import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import torch
from transformers import  BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular
from library import Graph, GraphLib
from utils import print_util as pu
from fine_tune_flexibert import finetune


def training(hash_model,task_ID)

	return "--model_name_or_path {} \
	  --task_name {} \
	  --do_train \
	  --do_eval \
	  --data_dir ../glue_data/{}/ \
	  --max_seq_length 128 \
	  --per_gpu_train_batch_size 32 \
	  --learning_rate 2e-5 \
	  --num_train_epochs 3.0 \
	  --output_dir {}".format('../models'+hash_model+'/',task_ID,task_ID,'../models'+hash_model+'/')


# Testing if bert-mini is in design_space_small
model_dict_bert_mini = {'l':4, 'a':[4]*4, 'f':[4*256]*4, 'h':[256]*4, 's':['sdp']*4}

graphLib = GraphLib.load_from_dataset('../dataset/dataset_small.json')
bert_mini_graph = graphLib.get_graph(model_dict=model_dict_bert_mini)

bert_mini_hash = str(bert_mini_graph.hash)

task1 = "SST-2"

task2 = "QNLI"


args_train = training(bert_mini_hash,task1)


metrics = finetune(args_train)


print(f'BERT mini accuracy on SST-2 {metrics['acc']:0.2f}')

args_train = training(bert_mini_hash,task2)

metrics = finetune(args_train)


print(f'BERT mini accuracy on QNLI {metrics['acc']:0.2f}')
