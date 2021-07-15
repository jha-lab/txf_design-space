
from roberta_pretraining import pretrain
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import os
import torch

import shlex
from library import Graph, GraphLib
import argparse

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


def main():
	"""Run pretraining
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model',
		metavar='',
		type=str,
		help='Model to pretrain',
		default='fnet_mini')

	args = parser.parse_args()


	graphLib = GraphLib.load_from_dataset('../dataset/dataset_test.json')

	seed = 1

	if args.model == 'fnet_mini':
		
		model_dict = {'l':4,'o':['l','l','l','l'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':['dft','dft','dft','dft']}

	elif args.model == 'fnet_tiny':

		model_dict =  { 'l': 2, 'o': ['l']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['dft']*2}

	elif args.model == 'convbert_mini':

		model_dict = {'l':4,'o':['c','c','c','c'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':[9,9,9,9]}

	elif args.model == 'convbert_tiny':

		model_dict= { 'l': 2, 'o': ['c']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': [9]*2}


	model_graph = graphLib.get_graph(model_dict=model_dict)[0]

	dataset = "cc_news"

	output_dir = "../../models/pretrained/"+str(model_graph.hash)+'/'

	args_train = training(dataset, seed, output_dir)

	# Forcing to train on single GPU
	#os.environ["CUDA_VISIBLE_DEVICES"]="0"

	metrics = pretrain(args_train, model_dict)

	print(f"MLM Loss on cc_news is {metrics['eval_loss']:0.2f}")

if __name__ == '__main__':
	
    main()
