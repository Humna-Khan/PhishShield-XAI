#!/bin/bash

# PhishShield-XAI Setup Script
echo "--- Starting PhishShield-XAI Environment Setup ---"

# 1. Python Backend Setup
echo "Setting up Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Backend Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Frontend Setup
if [ -d "web" ]; then
    echo "Setting up Frontend Dashboard..."
    cd web
    if [ -f "package.json" ]; then
        npm install
    fi
    cd ..
fi

# 3. Model Initialization
echo "Generating initial model artifacts..."
./venv/bin/python scripts/generate_artifacts.py

echo "--- Setup Complete! ---"
echo "You can now run the project using: ./run.sh"
