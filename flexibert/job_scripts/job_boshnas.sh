#!/bin/bash

# Sript to load GLUE dataset into cache using load_glue_datset.py 
# and then create a job script that trains a surrogate model for the
# given task

# Author : Shikhar Tuli


id="bdedhia"

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo
   echo -e "Syntax: source ${CYAN}job_creator_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"bdedhia\"${ENDC}] \t\t Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in

    -i | --id)
        shift
        id=$1
        shift
        ;;
    -h | --help)
       Help
       return 1;
       ;;
    *)
       echo "Unrecognized flag $1"
       return 1;
       ;;
esac
done  



cd ..

cd job_scripts/

job_file="run_boshnas.slurm"

# Create SLURM job script to train surrogate model
echo "#!/bin/bash
#SBATCH --job-name=boshnas                  # create a short name for your job 
#SBATCH --partition gpu
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=20                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --time=144:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=${id}@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..


python -u  run_boshnas.py --id ${id} --models_dir /scratch/gpfs/${id}/txf_design-space/models/ " > $job_file

sbatch $job_file
