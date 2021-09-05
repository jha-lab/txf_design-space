#!/bin/bash

# Script to train (either fine-tune or pre-train and then fine-tune)
# a given model.

# Author : Shikhar Tuli

task="glue"
cluster_gpu=""
pretrain="0"
autotune="0"
autotune_trials=""
model_hash=""
model_name_or_path=""
dataset_file=""
models_dir=""
epochs=""
output_dir=""

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Script to train (either fine-tune or pre-train and then fine-tune) a given model${ENDC}"
   echo
   echo -e "Syntax: source ${CYAN}./job_scripts/job_train_script.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-t${ENDC} | ${YELLOW}--task${ENDC} [default = ${GREEN}\"sst2\"${ENDC}] \t\t\t Name of GLUE task"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster_gpu${ENDC} [default = ${GREEN}\"2\"${ENDC}] \t\t\t Number of GPUs per cluster"
   echo -e "${YELLOW}-p${ENDC} | ${YELLOW}--pretrain${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To pre-train the given model"
   echo -e "${YELLOW}-a${ENDC} | ${YELLOW}--autotune${ENDC} [default = ${GREEN}\"0\"${ENDC}] \t\t To autotune the training recipe"
   echo -e "${YELLOW}-l${ENDC} | ${YELLOW}--autotune_trials${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Number of trials for autotuning"
   echo -e "${YELLOW}-m${ENDC} | ${YELLOW}--model_hash${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Model hash"
   echo -e "${YELLOW}-n${ENDC} | ${YELLOW}--model_name_or_path${ENDC} [default = ${GREEN}\"\"${ENDC}] \t Model path for fine-tuning"
   echo -e "${YELLOW}-d${ENDC} | ${YELLOW}--dataset_file${ENDC} [default = ${GREEN}\"\"${ENDC}] \t Path to the dataset file"
   echo -e "${YELLOW}-r${ENDC} | ${YELLOW}--models_dir${ENDC} [default = ${GREEN}\"\"${ENDC}] \t Path to the models directory"
   echo -e "${YELLOW}-e${ENDC} | ${YELLOW}--epochs${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t\t Number of epochs for fine-tuning"
   echo -e "${YELLOW}-o${ENDC} | ${YELLOW}--output_dir${ENDC} [default = ${GREEN}\"\"${ENDC}] \t\t Output directory to save fine-tuned result"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t\t Call this help message"
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
    -p | --pretrain)
        shift
        pretrain=$1
        shift
        ;;
    -a | --autotune)
        shift
        autotune=$1
        shift
        ;;
    -l | --autotune_trials)
        shift
        autotune_trials=$1
        shift
        ;;
    -m | --model_hash)
        shift
        model_hash=$1
        shift
        ;;
    -n | --model_name_or_path)
        shift
        model_name_or_path=$1
        shift
        ;;
    -d | --dataset_file)
        shift
        dataset_file=$1
        shift
        ;;
    -r | --models_dir)
        shift
        models_dir=$1
        shift
        ;;
    -e | --epochs)
        shift
        epochs=$1
        shift
        ;;
    -o | --output_dir)
        shift
        output_dir=$1
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

job_file="./job_${model_hash}_.slurm"
mkdir -p "./job_scripts/${task}/"

# Load GLUE dataset using the internet
if [[ $task == "glue" ]]
then
    python load_all_glue_datasets.py
else
    python load_glue_dataset.py --task $task
fi

cd "./job_scripts/${task}/"

# Create SLURM job script to train surrogate model
echo "#!/bin/bash" >> $job_file
echo "#SBATCH --job-name=flexibert_${task}_${model_hash}  # create a short name for your job" >> $job_file
echo "#SBATCH --partition ${partition}                     # parition to run job" >> $job_file
echo "#SBATCH --nodes=1                                   # node count" >> $job_file
echo "#SBATCH --ntasks=1                                  # total number of tasks across all nodes" >> $job_file
echo "#SBATCH --cpus-per-task=20                          # cpu-cores per task (>1 if multi-threaded tasks)" >> $job_file
echo "#SBATCH --mem-per-cpu=8G                            # memory per cpu-core (4G is default)" >> $job_file
echo "#SBATCH --gres=${cluster_gpu}                       # number of gpus per node" >> $job_file
echo "#SBATCH --time=144:00:00                            # total run time limit (HH:MM:SS)" >> $job_file
echo "" >> $job_file
echo "module purge" >> $job_file
echo "module load anaconda3/2020.7" >> $job_file
echo "conda activate txf_design-space" >> $job_file
echo "" >> $job_file
echo "cd ../../" >> $job_file
echo "" >> $job_file
if [[ $pretrain == "1" ]]
then
    echo "python pretrain_flexibert.py --model_hash ${model_hash} \
        --output_dir ${model_name_or_path} \
        --dataset_file ${dataset_file} \
        --id ${id}" >> $job_file
    echo "wait" >> $job_file
fi
if [[ $autotune == "0" ]]
then
    if [[ $task == "glue" ]]
    then
        echo "python glue_score.py --model_hash ${model_hash} \
            --id ${id} \
            --models_dir ${models_dir} " >> $job_file
    else
        echo "python finetune_flexibert.py --model_name_or_path ${model_name_or_path} \
            --task_name ${task} \
            --id ${id} \
            --do_train \
            --do_eval \
            --save_total_limit 2 \
            --max_sequence_length 128 \
            --per_device_train_batch_size 64 \
            --load_best_model_at_end \
            --learning_rate 2e-5 \
            --num_train_epochs ${epochs} \
            --overwrite_output_dir \
            --output_dir ${output_dir}" >> $job_file
    fi
else
    if [[ $task == "glue" ]]
    then
        echo "python glue_score.py --model_hash ${model_hash} \
            --id ${id} \
            --models_dir ${models_dir} \
            --autotune \
            --autotune_trials ${autotune_trials}" >> $job_file
    else
        echo "python finetune_flexibert.py --model_name_or_path ${model_name_or_path} \
            --task_name ${task} \
            --id ${id} \
            --do_train \
            --do_eval \
            --autotune \
            --autotune_trials ${autotune_trials} \
            --save_total_limit 2 \
            --max_sequence_length 128 \
            --per_device_train_batch_size 64 \
            --load_best_model_at_end \
            --learning_rate 2e-5 \
            --num_train_epochs ${epochs} \
            --overwrite_output_dir \
            --output_dir ${output_dir}" >> $job_file
    fi
fi

sbatch $job_file
