"""
Training script for Code Generator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from models.config import TransformerConfig
from models.transformer import CodeTransformer
from models.tokenizer import CodeTokenizer
from data_loader import CodeDataLoader
from utils import create_directories, save_checkpoint, print_model_info, Logger, get_device

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        
        # Create source and target sequences
        src = batch[:, :-1]
        tgt = batch[:, :-1]
        target = batch[:, 1:]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, model.config.vocab_size), target.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            
            src = batch[:, :-1]
            tgt = batch[:, :-1]
            target = batch[:, 1:]
            
            output = model(src, tgt)
            loss = criterion(output.reshape(-1, model.config.vocab_size), target.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    config = TransformerConfig()
    device = get_device()
    
    # Setup
    create_directories(config)
    logger = Logger(os.path.join(config.log_dir, "training.log"))
    writer = SummaryWriter(config.log_dir)
    
    logger.log(f"Device: {device}")
    logger.log(f"Config: {config.__dict__}")
    
    # Create model
    model = CodeTransformer(config).to(device)
    print_model_info(model)
    
    # Create tokenizer
    tokenizer = CodeTokenizer(config.vocab_size)
    tokenizer.save(os.path.join(config.data_dir, "tokenizer.json"))
    
    # Load data
    logger.log("Loading data...")
    data_loader = CodeDataLoader(config.data_dir, tokenizer, config.batch_size, config.max_seq_length)
    train_loaders = data_loader.load_all_languages()
    
    if not train_loaders:
        logger.log("No training data found. Please add code files to data/ directory")
        return
    
    logger.log(f"Loaded {len(train_loaders)} language datasets")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        logger.log(f"\nEpoch {epoch+1}/{config.num_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        # Train on all languages
        for language, train_loader in train_loaders.items():
            logger.log(f"Training on {language}...")
            batch_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1
            
            writer.add_scalar(f"train_loss/{language}", batch_loss, global_step)
        
        avg_loss = epoch_loss / num_batches
        logger.log(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        writer.add_scalar("train_loss/average", avg_loss, epoch)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, config)
            logger.log("✓ Best model saved")
    
    writer.close()
    logger.log("\nTraining completed!")

if __name__ == "__main__":
    main()
