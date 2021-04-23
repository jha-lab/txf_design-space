#!/bin/bash

#SBATCH --job-name=txf_design-space         # create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=32                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=23G                   # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:tesla_v100:0             # number of gpus per node
#SBATCH --time=24:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

python generate_library.py --design_space_file '../design_space/design_space.yaml' \
	--dataset_file '../dataset/dataset.json' \
	--embedding_size 16 \
	--n_jobs 24

