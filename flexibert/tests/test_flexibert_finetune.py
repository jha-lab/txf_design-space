import sys
sys.path.append('../../transformers/src/')
sys.path.append('../../embeddings/')
sys.path.append('../../flexibert/')
import os
import torch
from transformers import  BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular
from library import Graph, GraphLib
from utils import print_util as pu
from finetune_flexibert import finetune
import shlex

def training(hash_model,task_ID):

	a = "--model_name_or_path {} \
	--task_name {} \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
    --overwrite_output_dir \
	--output_dir {}".format('../../models/pretrained/'+hash_model+'/',task_ID,'../../models/'+task_ID+'/'+hash_model+'/')

	return shlex.split(a)


# Testing if bert-mini is in design_space_small
model_dict_bert_mini = {'l':4, 'a':[4]*4, 'f':[4*256]*4, 'h':[256]*4, 's':['sdp']*4}

graphLib = GraphLib.load_from_dataset('../../dataset/dataset_small.json')
bert_mini_graph = graphLib.get_graph(model_dict=model_dict_bert_mini)

bert_mini_hash = str(bert_mini_graph.hash)

task1 = "sst2"


args_train = training(bert_mini_hash,task1)

# Forcing to train on single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

metrics = finetune(args_train)

print(f"BERT mini accuracy on SST-2 {metrics['eval_accuracy']: 0.2f}")
