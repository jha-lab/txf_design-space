#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

module load anaconda3

conda create --name txf_design-space pytorch torchvision torchaudio cudatoolkit=10.2 --channel pytorch

conda activate txf_design-space

# Not needed for editable install
# conda install -c huggingface transformers

# Editable installation from source
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

# Add other packages
conda install pyyaml

## TODO: create environment.yaml file or requirements.txt

# Check installation
cd ..
python check_install.py

