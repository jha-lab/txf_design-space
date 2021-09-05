#!/bin/bash
#SBATCH --job-name=boshnas_homogeneous                  # create a short name for your job 
#SBATCH --partition gpu
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=10                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --time=144:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..


python -u  run_boshnas.py --id stuli --models_dir /scratch/gpfs/stuli/txf_design-space/models_homogeneous/ 
