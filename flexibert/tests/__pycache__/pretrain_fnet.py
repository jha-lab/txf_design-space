
from roberta_pretraining import pretrain
import sys
sys.path.append('../../transformers/src/')
sys.path.append('../../embeddings/')
sys.path.append('../../flexibert/')
import os
import torch
from utils import print_util as pu
import shlex





def training(dataset_name):

	a = "--output_dir ../../models/pretrained/ \
	--dataset_name {} \
	--do_train \
	--do_eval \
	--max_seq_length 512 \
	--line_by_line True \
	--ngpu 1 \
	--per_gpu_train_batch_size 32 \
    --overwrite_output_dir \
    ".format(dataset_name)

	return shlex.split(a)

dataset = "cc_news"

model_dict = {'l':4,'o':['l','l','l','l'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':['dft','dft','dft','dft']}

args_train = training(dataset)

# Forcing to train on single GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

metrics = pretrain(args_train, model_dict)

print(f"Accuracy on cc_news is {metrics['eval_accuracy']:0.2f}")
