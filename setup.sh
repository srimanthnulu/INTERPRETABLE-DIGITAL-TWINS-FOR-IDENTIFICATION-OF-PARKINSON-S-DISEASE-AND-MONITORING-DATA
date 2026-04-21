#!/bin/bash

# Quick Start Setup Script for Parkinson's Detection System
# This script helps you get started quickly

echo "=========================================="
echo "Parkinson's Detection System Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
    echo "Activate it with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
    echo ""
fi

# Install dependencies
read -p "Install required packages? (y/n): " install_deps
if [ "$install_deps" = "y" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Dependencies installed."
    echo ""
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/healthy/spiral
mkdir -p data/healthy/wave
mkdir -p data/healthy/handwriting
mkdir -p data/parkinsons/spiral
mkdir -p data/parkinsons/wave
mkdir -p data/parkinsons/handwriting
mkdir -p results
mkdir -p patient_records
echo "Directories created."
echo ""

# Create __init__.py files
echo "Creating Python package files..."
touch models/__init__.py
touch utils/__init__.py
touch app/__init__.py
echo "Package files created."
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your dataset images in the data/ directories"
echo "2. Train the model: python train.py --epochs 20"
echo "3. Launch the app: python app/gradio_app.py"
echo ""
echo "For detailed instructions, see README.md"
echo ""
