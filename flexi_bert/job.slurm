#!/bin/bash

#SBATCH --job-name=txf_design-space         # create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks-per-node=1                  	    # total number of tasks across all nodes
#SBATCH --mem=23G
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=00:20:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=bdedhia@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

python test_flexibert_finetune.py

