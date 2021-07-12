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
import subprocess
import json
import shlex

def training(model_hash, task):

	model_name_or_path = '../../models/pretrained/' + model_hash + '/'

	a = f'--model_name_or_path {model_name_or_path} \
        --task_name {task} \
        --do_train \
        --do_eval \
        --save_total_limit 2 \
        --max_seq_length 128 \
        --per_device_train_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        --output_dir {"../../models/" + task + "_test/" + model_hash + "/"}'

	return shlex.split(a), a


# Testing if bert-mini is in design_space_small
# model_dict_bert_mini = {'l':4, 'o': ['sa']*4, 'n':[4]*4, 'f':[[4*256]]*4, 'h':[256]*4, 'p':['sdp']*4}
bert_model_hash = str('19da7de67e79753a3caa5758d176f63c')

# graphLib = GraphLib.load_from_dataset('../../dataset/old/dataset_small.json')
# bert_mini_graph = graphLib.get_graph(model_hash=bert_mini_hash)

task1 = "sst2"

args_train, args_raw = training(bert_model_hash, task1)

# Forcing to train on single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Test finetune function
metrics = finetune(args_train)

# Test CLI command
stdout = subprocess.check_output(f'python ../finetune_flexibert.py {args_raw}', shell=True, text=True)

with open('../../models/' + task1 + '_test/' + bert_model_hash + '/all_results.json', 'r') as json_file:
        metrics_json = json.load(json_file) 

print(f"BERT mini accuracy on SST-2: {metrics['eval_accuracy']: 0.2f}")
print(f"BERT mini accuracy on SST-2 from JSON: {metrics['eval_accuracy']: 0.2f}")
