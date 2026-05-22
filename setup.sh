#!/bin/bash

echo "🌽 Setting up Hyperspectral Corn Analysis Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the Streamlit app:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run app/app.py"
echo ""
echo "To run Jupyter notebooks:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: jupyter notebook"
