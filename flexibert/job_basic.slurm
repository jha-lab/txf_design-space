#!/bin/bash

#SBATCH --job-name=txf_design-space         # create a short name for your job
#SBATCH --nodes=1                  	    	  # node count
#SBATCH --ntasks=1                  	      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4			                  # number of gpus per node
#SBATCH --time=00:20:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                   # send email when job begins
#SBATCH --mail-type=end                     # send email when job ends
#SBATCH --mail-user=stuli@princeton.edu


module purge
module load anaconda3/2020.7
conda activate txf_design-space

python run_surrogate_model.py --task sst2