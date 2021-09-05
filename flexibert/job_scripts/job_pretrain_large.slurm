#!/bin/bash

#SBATCH --job-name=flexibert_large   # create a short name for your job
#SBATCH --partition gpu-ee
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=20                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=240G                          # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2                        # number of gpus per node
#SBATCH --time=144:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..

python pretrain_flexibert_large.py --output_dir /scratch/gpfs/stuli/txf_design-space/models_large/pretrained/ --id stuli

