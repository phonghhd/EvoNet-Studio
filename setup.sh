#!/bin/bash
echo "Setting up EvoNet-Studio Virtual Environment..."

# Ensure we have a venv directory
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate environment
source venv/bin/activate

# Create a local temporary directory to prevent filling up the system /tmp
mkdir -p tmp_pip
export TMPDIR=$(pwd)/tmp_pip

echo "Upgrading pip..."
pip install --no-cache-dir --upgrade pip

echo "Installing dependencies (Optimized Mode: No Cache)..."
# Using --no-cache-dir prevents pip from keeping large cached .whl files (like PyTorch)
pip install --no-cache-dir -r requirements.txt

# Clean up
rm -rf tmp_pip

echo "Setup complete! To run the studio:"
echo "source venv/bin/activate"
echo "python app.py"
