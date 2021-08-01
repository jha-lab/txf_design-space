#!/bin/bash
#SBATCH --job-name=prepare_data        # create a short name for your job
#SBATCH --partition gpu-100
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=05:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=bdedhia@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..

python prepare_pretrain_dataset.py
