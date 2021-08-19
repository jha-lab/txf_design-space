#!/bin/sh

# Script to install required packages in conda for GPU setup
# Author : Shikhar Tuli

cluster="tiger"

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
ENDC='\033[0m'

Help()
{
   # Display Help
   echo -e "${CYAN}Sript to setup conda environment" 
   echo
   echo -e "Syntax: source ${CYAN}env_setup.sh${ENDC} [${YELLOW}flags${ENDC}]"
   echo "Flags:"
   echo -e "${YELLOW}-c${ENDC} | ${YELLOW}--cluster${ENDC} [default = ${GREEN}\"tiger\"${ENDC}] \t Selected cluster - adroit or tiger"
   echo -e "${YELLOW}-h${ENDC} | ${YELLOW}--help${ENDC} \t\t\t\t Call this help message"
   echo
}

while [[ $# -gt 0 ]]
do
case "$1" in
    -c | --cluster)
        shift
        cluster=$1
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
  module load anaconda3
  conda create --name txf_design-space pytorch torchvision torchaudio cudatoolkit=10.2 --channel pytorch --channel nvidia
elif [[ $cluster == "tiger" ]]
then
  module load anaconda3
  conda create --name txf_design-space pytorch torchvision torchaudio cudatoolkit=10.2 --channel pytorch --channel nvidia
elif [[ $cluster == "della" ]]
then
	module load anaconda3/2020.11
	conda create --name txf_design-space pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia
else
	echo "Unrecognized cluster"
	return 1
fi

conda activate txf_design-space

# Not needed for editable install
# conda install -c huggingface transformers

# Editable installation from source
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
pip install torch-dct
cd ..

# Install datasets
git clone https://github.com/huggingface/datasets.git
cd datasets/
pip install -e .
cd ..

# Add other packages and enabling extentions
conda install -c conda-forge tqdm ipywidgets matplotlib scikit-optimize
jupyter nbextension enable --py widgetsnbextension
conda install -c anaconda scipy cython
conda install pyyaml
conda install pandas
conda install -c plotly plotly

# Conda prefers pip packages in the end
pip install grakel
pip install datasets
pip install networkx
pip install tabulate

## TODO: create environment.yaml file or requirements.txt

# Check installation
python check_install.py
