#!/usr/bin/bash

python3 -m venv smudges_env                                      # Create a new virtual environment
source activate smudges_env/bin/activate                         # Activate virtual environment
pip install numpy==1.19.1                                        # Install NumPy
pip install pandas==0.25.1                                       # Install pandas
pip install scikit-image==0.17.2                                 # Install skimage
pip install torch==1.6.0                                         # Install PyTorch
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh   # Installs Light Version of LaTeX

