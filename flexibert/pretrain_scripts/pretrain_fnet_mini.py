
from roberta_pretraining import pretrain
import sys
sys.path.append('../../transformers/src/')
sys.path.append('../../embeddings/')
sys.path.append('../../flexibert/')

import os
import torch
from utils import print_util as pu
import shlex
from library import Graph, GraphLib

graphLib = GraphLib.load_from_dataset('../dataset/dataset_test.json')

import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)


def training(dataset_name, seed, output_dir):

	a = "--dataset_name {} \
	--seed {} \
	--do_train \
	--do_eval \
	--max_seq_length 512 \
	--line_by_line True \
	--per_gpu_train_batch_size 64 \
	--adam_epsilon 1e-6\
	--learning_rate 1e-4\
	--warmup_steps 10000\
	--lr_scheduler_type linear\
    --overwrite_output_dir \
    --output_dir {} \
    ".format(dataset_name, seed, output_dir)

	return shlex.split(a)


seed = 1

model_dict = {'l':4,'o':['l','l','l','l'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':['dft','dft','dft','dft']}

fnet_mini_graph = graphLib.get_graph(model_dict=model_dict)[0]

dataset = "cc_news"

output_dir = "../../models/pretrained/"+str(fnet_mini_graph.hash)+'/'

args_train = training(dataset, seed, output_dir)

# Forcing to train on single GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

metrics = pretrain(args_train, model_dict)

print(f"MLM Accuracy on cc_news is {metrics['eval_accuracy']:0.2f}")
