"""
Custom Tokenizer for Code
"""

import json
import os
from collections import Counter
from typing import List, Tuple

class CodeTokenizer:
    """Tokenizer for programming code"""
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3,
        '<NEWLINE>': 4,
        '<INDENT>': 5,
        '<DEDENT>': 6,
    }
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.token2id = self.SPECIAL_TOKENS.copy()
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.current_id = len(self.SPECIAL_TOKENS)
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts"""
        counter = Counter()
        
        for text in texts:
            tokens = self.preprocess_text(text)
            counter.update(tokens)
        
        # Add most common tokens
        for token, freq in counter.most_common(self.vocab_size - len(self.SPECIAL_TOKENS)):
            if freq >= min_freq:
                self.token2id[token] = self.current_id
                self.id2token[self.current_id] = token
                self.current_id += 1
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into tokens"""
        # Replace special characters
        text = text.replace('\n', ' <NEWLINE> ')
        text = text.replace('\t', ' <INDENT> ')
        
        # Split by spaces and punctuation (but keep them)
        tokens = []
        current_token = ""
        
        for char in text:
            if char in " \n\t":
                if current_token:
                    tokens.append(current_token)
                current_token = ""
            elif char in "(){}[]<>.,;:=+-*/%&|^!~?":
                if current_token:
                    tokens.append(current_token)
                tokens.append(char)
                current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.preprocess_text(text)
        return [self.token2id.get(token, self.token2id['<UNK>']) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id2token.get(id, '<UNK>') for id in ids]
        text = ' '.join(tokens)
        text = text.replace(' <NEWLINE> ', '\n')
        text = text.replace(' <INDENT> ', '\t')
        return text
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'token2id': self.token2id,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'vocab_size': self.vocab_size
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.token2id = data['token2id']
        self.id2token = {int(k): v for k, v in data['id2token'].items()}
        self.vocab_size = data['vocab_size']
        self.current_id = max(self.id2token.keys()) + 1
