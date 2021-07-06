
from roberta_pretraining import pretrain
import sys
sys.path.append('../../transformers/src/')
sys.path.append('../../embeddings/')
sys.path.append('../../flexibert/')
import os
import torch
from utils import print_util as pu
import shlex





def training(dataset_name,model_dict):

	a = "--model_name_or_path {} \
	--task_name {} \
	--do_train \
	--do_eval \
	--per_gpu_train_batch_size 32 \
    --overwrite_output_dir \
	--output_dir {}"..format(dataset_name,model_dict)

	return shlex.split(a)



dataset = "cc_news"
model_dict = {'l': 4, 'o': ['l', 'l', 'l', 'l'], 'h': [256, 256, 256, 256], 'n': [4, 4, 4, 4], 'f': [[1024], [1024], [1024], [1024]], 'p': ['dft', 'dft', 'dft', 'dft']}


args_train = training(bert_mini_hash,task1)

# Forcing to train on single GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

metrics = pretrain(args_train)

print(f"Accuracy on cc_news is {metrics['eval_accuracy']: 0.2f}")
