#!/usr/bin/bash

# Create a new virtual environment
python3 -m venv smudges_env

# Activate virtual environment
source activate smudges_env/bin/activate

# Install Python Packages
pip install numpy==1.19.1
pip install pandas==0.25.1
pip install scikit-image==0.17.2
pip install torch==1.6.0
pip install torchvision==0.2.2

# Install Light Version of LaTeX
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh

