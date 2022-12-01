# FlexiBERT: an Exploratory Study of the Transformer Design Space

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

FlexiBERT is a tool which can be used to generate and evaluate different Transformer architectures on diverse NLP tasks.
This repository has been forked from [huggingface/transformers](https://github.com/huggingface/transformers) and then expanded to incorporate more heterogenous Transformer architectures.

## Table of Contents
- [Environment setup](#environment-setup)
  - [Clone this repository and initialize sub-modules](#clone-this-repository-and-initialize-sub-modules)
  - [Setup python environment](#setup-python-environment)
- [Replicating results](#replicating-results)
- [Pre-trained models](#pre-trained-models)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

### Clone this repository and initialize sub-modules

```shell
git clone https://github.com/jha-lab/txf_design-space.git
cd ./txf_design-space/
git submodule init
git submodule update
```

### Setup python environment  

The python environment setup is based on conda. The script below creates a new environment named `txf_design-space`:
```shell
source env_step.sh
```
To install using pip, use the following command:
```shell
pip install -r requirements.txt
```
To test the installation, you can run:
```shell
python check_install.py
```
All training scripts use bash and have been implemented using [SLURM](https://slurm.schedmd.com/documentation.html). This will have to be setup before running the experiments.

## Replicating results

### Specify the design space

For this, `.yaml` files can be used. Examples are given in the `dataset/` directory. For the experiments in the paper, `design_space/design_space_test.yaml` was used.

### Generate the graph library

This can be done in mutiple steps in the hierarchy. From the given design space: `design_space/design_space_test.yaml`, the graph library is created at `dataset/dataset_test_bn.json` with neighbors decided using _biased overlap_ as follows:
```shell
cd embeddings/
python generate_library.py --design_space ../design_space/design_space_test.yaml --dataset_file ../dataset/dataset_test_bn.json --layers_per_stack 2
cd ../
```
Other flags can also be used to control the graph library generation (check using `python embeddings/generate_library.py --help`).

### Prepare pre-training and fine-tuning datasets

Run the following scripts:
```shell
cd flexibert/
python prepare_pretrain_dataset.py
python save_roberta_tokenizer.py
python load_all_glue_datasets.py
python tokenize_glue_datasets.py
cd ../
```

### Run BOSHNAS

For the selected graph library, run BOSHNAS with the following command:
```shell
cd flexibert/
python run_boshnas.py
cd ../
```
Other flags can be used to control the training procedure (check using `python flexibert/run_boshnas.py --help`). This script uses the SLURM scheduler over mutiple compute nodes in a cluster (each cluster assumed to have 2 GPUs, this can be changed in code). SLURM can also be used in scenarios where distributed nodes are not available.

### Generate graph library for next level of hierarchy

To generate a graph library with `layers_per_stack=1` from the best models in the first level, use the following command:
```shell
cd flexibert/
python hierarchical.py --old_dataset_file ../dataset/dataset_test_bn.json --new_dataset_file ../dataset/dataset_test_bn_2.json --old_layers_per_stack 2 --new_layers_per_stack 1 
cd ../
```
This saves a new graph library for the next level of the hierarchy. Heterogeneous feed-forward stacks can also be generated using the flag `--heterogeneous_feed_forward`.

For this new graph library, BOSHNAS can be run again to get the nest set of best-performing models.

## Pre-trained models

The pre-trained models are accessible [here](https://drive.google.com/drive/folders/1-0orzWsHtITO6ltyhvCY2Yh5sX19Smom?usp=sharing). 

To use the downloaded FlexiBERT-Mini model:
```python
flexibert_mini = FlexiBERTModel.from_pretrained('./models/flexibert_mini/')
```

To instantiate a model in the FlexiBERT design space, create a model dictionary and generate a model configuration:
```python
model_dict = {'l': 4, 'o': ['sa', 'sa', 'l', 'l'], 'h': [256, 256, 128, 128], 'n': [2, 2, 4, 4],
      'f': [[512, 512, 512], [512, 512, 512], [1024], [1024]], 'p': ['sdp', 'sdp', 'dct', 'dct']}
flexibert_mini_config = FlexiBERTConfig()
flexibert_mini_config.from_model_dict(model_dict)
flexibert_mini = FlexiBERTModel(flexibert_mini_config)
```

You can also use the FlexiBERT 2.0 `hetero` model dictionary format (paper under review). To transfer weights to another model within the design space (both should be in standard or `hetero` format):
```python
model_dict = {'l': 2, 'o': ['sa', 'sa'], 'h': [128, 128], 'n': [2, 2],
      'f': [[512], [512]], 'p': ['sdp', 'sdp']}
bert_tiny_config = FlexiBERTConfig()
bert_tiny_config.from_model_dict(model_dict)
bert_tiny = FlexiBERTModel(bert_tiny_config, transfer_mode='RP')

# Implement fine-grained knowledge transfer using random projections
bert_tiny.load_model_from_source(flexibert_mini)
```

We will be adding more pre-trained models so stay tuned!

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{tuli2022jair,
      title={{FlexiBERT}: Are Current Transformer Architectures too Homogeneous and Rigid?}, 
      author={Tuli, Shikhar and Dedhia, Bhishma and Tuli, Shreshth and Jha, Niraj K.},
      year={2022},
      eprint={2205.11656},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
