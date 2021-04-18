# Transformer Design Space exploration
COS 484 project on Transformer design space exploration.

## Environment Setup

Run the following command to setup conda environment and implement a basic test:
```
source env_step.sh
```

## Architecture hyper-parameters

The model architectures are restricted in the following ranges for the respective hyper-parameters (we follow the same terminology as the [schuBERT](https://www.aclweb.org/anthology/2020.acl-main.250.pdf) paper):

- `dataset` in the GLUE benchmark tasks: [CoLA, SST-2, MRPC, STS-B, QQP, MNLI-mm, QNLI, RTE, WNLI]
- `h`, the hidden size in: [128, **256**, 512]
- `a`, the number of attention heads in: [2, **4**, 8]
- `l`, the number of encoder layers in: [2, **4**, 6]
- `f`, the inner-layer dimensionality of the feed-forward layer in: [512, 1024, **2048**, 4096]
- `s`, the similarity metric in: [**scaled dot-product (sdp)**, weighted multiplicative attention (wma)]

*The hyper-parameter values in bold text correspond to the design choices in BERT-mini. Nine pre-trained models within this design space can be found at this repo*: [google-research/bert](https://github.com/google-research/bert).

Every model will be represented by a dictionary (referred to by the variable `model_dict`). An example dictionary for BERT-mini is as follows:

```
model_dict = {'l': 4, 'h': [256, 256, 256, 256], 'a': [4, 4, 4, 4], 'f': [2048, 2048, 2048, 2048], 's': ['sdp', 'sdp', 'sdp', 'sdp']}
```

This dictionary will be converted to a pytorch model for training, where the weights would be transferred from the 'nearest' pre-trained model. This modular simulator is a wrapper over the [huggingface/transformers](https://github.com/huggingface/transformers) repo.

The computation of the 'nearest' transformer is done using graph embeddings for every transformer's computational graph in this design space. 
