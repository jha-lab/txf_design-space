
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
#logging.disable(logging.INFO)
#logging.disable(logging.WARNING)


def get_training_args(seed, output_dir):

	a = "--seed {} \
	--do_train \
	--do_eval \
	--max_seq_length 512 \
	--per_gpu_train_batch_size 8 \
	--num_train_epochs 20.0 \
	--adam_epsilon 1e-6 \
	--learning_rate 1e-4 \
	--save_total_limit 2 \
	--warmup_steps 10000 \
	--lr_scheduler_type linear \
	--output_dir {} \
    --overwrite_output_dir \
        ".format( seed, output_dir)

	return shlex.split(a)


def test():
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

	elif args.model == 'fnet_2_256':

		model_dict = { 'l': 2, 'o': ['l']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['dft']*2}

	elif args.model == 'fnet_4_128':

		model_dict = { 'l': 4, 'o': ['l']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['dft']*4}


	elif args.model == 'convbert_2_256':

		model_dict = { 'l': 2, 'o': ['c']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': [9]*2}

	elif args.model == 'convbert_4_128':

		model_dict = { 'l': 4, 'o': ['c']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': [9]*4}


	model_graph = graphLib.get_graph(model_dict=model_dict)[0]


	output_dir = "../models/pretrained/"+str(model_graph.hash)+'/'

	args_train = get_training_args(seed, output_dir)

	# Forcing to train on single GPU
	#os.environ["CUDA_VISIBLE_DEVICES"]="0"

	metrics = pretrain(args_train, model_dict)

	print(f"MLM Loss on cc_news is {metrics['eval_loss']:0.2f}")


def main():
	"""Pretraining front-end function"""
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--output_dir',
		metavar='',
		type=str,
		help='path to save the pretrained model')
	parser.add_argument('--id',
		metavar='',
		type=str,
		help='PU-NetID that is used to run slurm commands',
		default='stuli')

	args = parser.parse_args()

	# model_dict = {'l': 24, 'o': ['sa']*12 + ['l']*12, 'h': [1024]*12 + [512]*12, 'n': [8]*12 + [16]*12, \
	# 	'f': [[2048]*3]*12 + [[4096]]*12, 'p': ['sdp']*12 + ['dct']*12}

	model_dict = {'l': 4, 'h': [256, 256, 128, 128], 'n': [2, 2, 4, 4], 'o': ['sa', 'sa', 'l', 'l'], \
		'f': [[512, 512, 512], [512, 512, 512], [1024], [1024]], 'p': ['sdp', 'sdp', 'dct', 'dct']}

	seed = 1
	args_train = get_training_args(seed, args.output_dir)

	metrics = pretrain(args_train, model_dict, args.id)


if __name__ == '__main__':
    main()
