#!/bin/bash

# Sript to load GLUE dataset into cache using load_glue_datset.py 
# and then create a job script that trains a surrogate model for the
# given task

# Author : Shikhar Tuli

task="sst2"
cluster="tiger"
id="stuli"

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Sript to load GLUE dataset into cache using load_glue_datset.py" 
   echo -e "and then create a job script that trains a surrogate model for the"
   echo -e "given task.${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}job_creator_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-t${ENDC} | ${YELLOW}--task${ENDC} [default = ${GREEN}\"sst2\"${ENDC}] \t\t Name of GLUE task"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"tiger\"${ENDC}] \t Selected cluster - adroit or tiger"
   echo -e "${YELLOW}-i${ENDC} | ${YELLOW}--id${ENDC} [default = ${GREEN}\"stuli\"${ENDC}] \t\t Selected PU-NetID to email slurm updates"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -t | --task)
        shift
        task=$1
        shift
        ;;
    -c | --cluster)
        shift
        cluster=$1
        shift
        ;;
    -i | -id)
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

if [[ $cluster == "adroit" ]]
then
  cluster_gpu="gpu:tesla_v100:4"
elif [[ $cluster == "tiger" ]]
then
  cluster_gpu="gpu:4"
else
	echo "Unrecognized cluster"
	return 1
fi

job_file="job_surrogate_model_${task}.slurm"
surrogate_model_file="../dataset/surrogate_models/gp_${task}.pkl"

cd ..

# Load GLUE dataset using the internet
python load_glue_dataset.py --task $task

cd job_scripts/

# Create SLURM job script to train surrogate model
echo "#!/bin/bash
#SBATCH --job-name=flexibert_${task}        # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=${cluster_gpu}               # number of gpus per node
#SBATCH --time=48:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=${id}@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

cd ..

python run_surrogate_model.py --task ${task} --models_dir /scratch/gpfs/${id}/models/ --surrogate_model_file ${surrogate_model_file}" > $job_file

sbatch $job_file
