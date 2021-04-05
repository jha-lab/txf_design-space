#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

module load anaconda3

conda create --name txf_design-space pytorch torchvision torchaudio cudatoolkit=10.2 --channel pytorch

conda activate txf_design-space

conda install -c huggingface transformers
