#!/bin/bash
# AI Code Generator Launcher for Linux/Mac

echo ""
echo "============================================================"
echo "AI Code Generator - Neural Network"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

# Check if dependencies are installed
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Choose an option:"
echo "1. Train the model"
echo "2. Generate code"
echo "3. Exit"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting training..."
        python3 train.py
        ;;
    2)
        echo ""
        echo "Starting code generation..."
        python3 inference.py
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please try again."
        ;;
esac
