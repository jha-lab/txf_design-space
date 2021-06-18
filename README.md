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
- [Architecture hyperparameters](#architecture-hyperparameters)
- [Transformer2vec embeddings](#transformer2vec-embeddings)
- [Replicating results](#replicating-results)
- [Colab](#colab)

## Environment setup

### Clone this repository

```
git clone https://github.com/shikhartuli/txf_design-space.git
cd txf_design-space
```
### Setup python environment  

```
source env_step.sh
```

## Architecture hyperparameters

The model architectures are restricted in the following ranges for the respective hyper-parameters (we follow the same terminology as the [schuBERT](https://www.aclweb.org/anthology/2020.acl-main.250.pdf) paper):

- `dataset` in the GLUE benchmark tasks: [CoLA, SST-2, MRPC, STS-B, QQP, MNLI-mm, QNLI, RTE, WNLI]
- `h`, the hidden size in: [128, **256**, 512]
- `a`, the number of attention heads in: [2, **4**, 8]
- `t`, the type of attention used in: [linear(l), **self attention (sa)**, dynamic convolution (c)]
- `l`, the number of encoder layers in: [2, **4**, 6]
- `f`, the inner-layer dimensionality of the feed-forward layer in: [512, **1024**, 2048, 4096]
- `nff`, the number of feedforward stacks in [**1**, 2, 3]
- `s`, the similarity metric in: [**scaled dot-product (sdp)**, weighted multiplicative attention (wma)]

*The hyperparameter values in bold text correspond to the design choices in BERT-mini. Nine pre-trained models within this design space can be found at this repo:* [google-research/bert](https://github.com/google-research/bert).

Every model will be represented by a model card (referred to by the a dictionary `model_dict`). An example dictionary for BERT-mini is as follows:

```
model_dict = {'l': 4, 'h': [256, 256, 256, 256], 'a': [4, 4, 4, 4], 'f': [1024, 1024, 1024, 1024], 's': ['sdp', 'sdp', 'sdp', 'sdp']}
```

This dictionary is converted to a pytorch model for training, where the weights are transferred from the 'nearest' pre-trained model. This modular simulator is a wrapper over the [huggingface/transformers](https://github.com/huggingface/transformers) repo. Implementation details can be found at `transformers/src/transformers/models/bert/modeling_modular_bert.py`.

For the purpose of the project, a smaller design space was considered instead. The details of the design space in consideration can be found at `design_space/design_space_small.yaml`.

## Transformer2vec embeddings

The Transformer2vec embeddings were trained using global similarities generated by the Weisfeiler-Lehman Graph Kernel. Using these similiarity values, Multi-Dimensional Scaling (MDS) was applied to generate embeddings of eight dimensions.

Every Transformer model is saved as a computational graph in the `GraphLib` class. Further, every entry in the list `GraphLib.library` is an object of the class `Graph`, which contains its model card (a dictionary as shown above), the adjacency matrix for its computational graph, a unique hash (used to load/store of checkpoints and also to check isomorphisms) and its embedding. To build a Graph Library and test the embedding for BERT-Mini, change the current directory to `embeddings/` and inside a python console:

```python
from library import GraphLib, Graph

graphLib = GraphLib(design_space='../design_space/design_space_small.yaml')

# Build a library of Transformers with increasing width
graphLib.build_library(increasing=True)

# Get Graph in the GraphLib object corresponding to BERT-Mini
model_dict = {'l': 4, 'h': [256]*4, 'a': [4]*4, 'f': [1024]*4, 's': ['sdp']*4}
print(graphLib.get_graph(model_dict)) 
```

## Replicating results

To replicate the results in the submitted report, a surrogate model will have to be trained over the design space of Transformer architectures. This can be done by running the file: `flexibert/run_surrogate_model.py`.

*For more details, on usage of the surrogate modeling code, use:* `python run_surrogate_model.py --help`.

For instance, to create the embeddings for the given design space and run an active learning framework to train a surrogate model over that design space for the `sst2` GLUE task:

```
cd embeddings
python generate_library.py --design_space_file '../design_space/design_space_small.yaml' --dataset_file '../dataset/dataset_small.json' --embedding_size 8

cd ..
cd flexibert
python save_pretrained_models.py
python load_glue_dataset.py --task sst2
python run_surrogate_model.py --task sst2 --models_dir ../models/ --surrogate_model_file ../dataset/surrogate_models/gp_sst2.pkl
```

This process has been automated as well. Once the dataset has been stored to `dataset/dataset_small.json` containing the Transformer `GraphLib` object, to run the surrogate modeling framework on a multi-GPU cluster node, use `flexibert/job_scripts/job_creator_script.sh`. Since the cluster nodes do not support internet connection, this script automatically downloads the given GLUE dataset before forwarding the slurm request.

*For more details on how to use this script, check:* `source flexibert/job_scripts/job_creator_script.sh --help`. 

Currently, these scripts only support running on **Adroit/Tiger clusters** at Princeton University. More information about these clusters and their usage can be found at the [Princeton Research Computing website](https://researchcomputing.princeton.edu/systems-and-services/available-systems).

## Colab

You can directly run tests on the generated dataset using a Google Colaboratory without needing to install anything on your local machine. Click "Open in Colab" below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shikhartuli/txf_design-space/blob/main/visualization/results.ipynb)
