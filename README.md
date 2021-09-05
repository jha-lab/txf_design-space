# FlexiBERT: an Exploratory Study of the Transformer Design Space

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

FlexiBERT is a tool which can be used to generate and evaluate different Transformer architectures on diverse NLP tasks.
This repository has been forked from [huggingface/transformers](https://github.com/huggingface/transformers) and then expanded to incorporate more heterogenous Transformer architectures.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup python environment](#setup-python-environment)
- [Replicating results](#replicating-results)

## Environment setup

### Clone this repository

```
git clone https://github.com/jha-lab/txf_design-space.git
cd txf_design-space
```

### Setup python environment  

The python environment setup is based on conda. The script below creates a new environment named `txf_design-space`:
```
source env_step.sh
```
To install using pip, use the following command:
```
pip install -r requirements.txt
```
To test the installation, you can run:
```
python check_install.py
```
All training scripts use bash and have been implemented using [SLURM](https://slurm.schedmd.com/documentation.html). This will have to be setup before running the experiments.

## Replicating results

### Specify the design space

For this, `.yaml` files can be used. Examples are given in the `dataset/` directory. For the experiments in the paper, `design_space/design_space_test.yaml` was used.

### Generate the graph library

This can be done in mutiple steps in the hierarchy. From the given design space: `design_space/design_space_test.yaml`, the graph library is created at `dataset/dataset_test_bn.json` with neighbors decided using _biased overlap_ as follows:
```
cd embeddings/
python generate_library.py --design_space ../design_space/design_space_test.yaml --dataset_file ../dataset/dataset_test_bn.json --layers_per_stack 2
cd ../
```
Other flags can also be used to control the graph library generation (check using `python embeddings/generate_library.py --help`).

### Prepare pre-training and fine-tuning datasets

Run the following scripts:
```
cd flexibert/
python prepare_pretrain_dataset.py
python save_roberta_tokenizer.py
python load_all_glue_datasets.py
python tokenize_glue_datasets.py
cd ../
```

### Run BOSHNAS

For the selected graph library, run BOSHNAS with the following command:
```
cd flexibert/
python run_boshnas.py
cd ../
```
Other flags can be used to control the training procedure (check using `python flexibert/run_boshnas.py --help`). This script uses the SLURM scheduler over mutiple compute nodes in a cluster (each cluster assumed to have 2 GPUs, this can be changed in code). SLURM can aso used in scenarios where distributed nodes are not available.

### Generate graph library for next level of hierarchy

To generate a graph library with `layers_per_stack=1` from the best models in the first level, use the following command:
```
cd flexibert/
python hierarchical.py --old_dataset_file ../dataset/dataset_test_bn.json --new_dataset_file ../dataset/dataset_test_bn_2.json --old_layers_per_stack 2 --new_layers_per_stack 1 
cd ../
```
This saves a new graph library for the next level of the hierarchy. Heterogeneous feed-forward stacks can also be generated using the flag `--heterogeneous_feed_forward`.

For this new graph library, BOSHNAS can be run again to get the nest set of best-performing models.
