
from roberta_pretraining import pretrain
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import os
import torch

import shlex
from library import Graph, GraphLib
from utils import graph_util
import argparse

import logging
#logging.disable(logging.INFO)
#logging.disable(logging.WARNING)


def get_training_args(seed, output_dir, local_rank):

	a = "--seed {} \
	--do_train \
	--do_eval \
	--max_seq_length 512 \
	--per_gpu_train_batch_size 64 \
	--max_steps 1000000 \
	--adam_epsilon 1e-6 \
	--adam_beta2 0.98 \
	--learning_rate 1e-4 \
	--weight_decay 0.01 \
	--save_total_limit 2 \
	--warmup_steps 10000 \
	--lr_scheduler_type linear \
	--output_dir {} \
    --overwrite_output_dir \
    --local_rank {} \
        ".format(seed, output_dir, local_rank)

	return shlex.split(a)


def test(args):
	"""Run pretraining
	"""
	graphLib = GraphLib.load_from_dataset('../dataset/dataset_test_bn.json')

	seed = 1

	if args.model_name == 'fnet_mini':
		
		model_dict = {'l':4,'o':['l','l','l','l'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':['dft','dft','dft','dft']}

	elif args.model_name == 'fnet_tiny':

		model_dict =  { 'l': 2, 'o': ['l']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['dft']*2}

	elif args.model_name == 'convbert_mini':

		model_dict = {'l':4,'o':['c','c','c','c'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':[9,9,9,9]}

	elif args.model_name == 'convbert_tiny':

		model_dict= { 'l': 2, 'o': ['c']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': [9]*2}

	elif args.model_name == 'fnet_2_256':

		model_dict = { 'l': 2, 'o': ['l']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['dft']*2}

	elif args.model_name == 'fnet_4_128':

		model_dict = { 'l': 4, 'o': ['l']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['dft']*4}

	elif args.model_name == 'convbert_2_256':

		model_dict = { 'l': 2, 'o': ['c']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': [9]*2}

	elif args.model_name == 'convbert_4_128':

		model_dict = { 'l': 4, 'o': ['c']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': [9]*4}

	elif args.model_name == 'bert_mini':

		model_dict = { 'l': 4, 'o': ['sa']*4, 'h': [256]*4, 'n': [4]*4, 'f': [[1024]]*4, 'p': ['sdp']*4}

	elif args.model_name == 'flexibert_mini':

		model_dict = { 'l': 4, 'o': ['sa', 'sa', 'l', 'l'], 'h': [256, 256, 128, 128], 'n': [2, 2, 4, 4], \
			'f': [[512, 512, 512], [512, 512, 512], [1024], [1024]], 'p': ['sdp', 'sdp', 'dct', 'dct']}

	elif args.model_name == 'flexibert_mini_no_second_order':

		model_dict = {'l': 2, 'h': [128, 128], 'n': [4, 4], 'o': ['sa', 'sa'], 'f': [[1024], [1024]], 'p': ['sdp', 'sdp']}

	elif args.model_name == 'flexibert_mini_no_heteroscedastic':

		model_dict = {'l': 4, 'h': [256, 256, 128, 128], 'n': [4, 4, 4, 4], 'o': ['l', 'l', 'sa', 'sa'], \
			'f': [[1024, 1024, 1024], [1024, 1024, 1024], [512, 512, 512], [512, 512, 512]], 'p': ['dct', 'dct', 'sdp', 'sdp']}

	elif args.model_name == 'flexibert_large':

		model_dict = { 'l': 24, 'o': ['sa']*12 + ['l']*12, 'h': [1024]*12 + [512]*12, 'n': [8]*12 + [16]*12, \
			'f': [[2048, 2048, 2048]]*12 + [[4096]]*12 , 'p': ['sdp']*12 + ['dct']*12}

	elif args.model_name == 'bert_base_hetero':

		model_dict = {'l': 12, 'o': [['sa_sdp_64']*12]*12, 'h': [768]*12, 'f': [[3072]]*12}

	if args.model_name.endswith('hetero'):
		model_graph = graph_util.model_dict_to_graph(model_dict)
		model_hash = graph_util.hash_graph(*model_graph)
		output_dir = '../models/pretrained/'+model_hash+'/'

	elif args.model_name != 'flexibert_large':
		model_graph = graphLib.get_graph(model_dict=model_dict)[0]
		output_dir = "../models/pretrained/"+str(model_graph.hash)+'/'


	else:
		model_graph = Graph(model_dict=model_dict, ops_list=None, compute_hash=True)
		output_dir = "../models/pretrained/"+str(model_graph.hash)+'/'

	args_train = get_training_args(seed, output_dir, args.local_rank)

	# Forcing to train on single GPU
	#os.environ["CUDA_VISIBLE_DEVICES"]="0"

	if args.model_name.endswith('hetero'):
		metrics, log_history, model = pretrain(args_train, model_dict)
	else:
		metrics = pretrain(args_train, model_dict)

	print(f"MLM Loss on cc_news is {metrics['eval_loss']:0.2f}")


def main(args):
	"""Pretraining front-end function"""
	graphLib = GraphLib.load_from_dataset(args.dataset_file)

	model_graph, _ = graphLib.get_graph(model_hash=args.model_hash)

	seed = 1
	args_train = get_training_args(seed, args.output_dir, args.local_rank)

	metrics = pretrain(args_train, model_graph.model_dict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Input parameters for pretraining',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_hash',
		metavar='',
		type=str,
		help='hash of the model to pretrain')
	parser.add_argument('--output_dir',
		metavar='',
		type=str,
		help='path to save the pretrained model')
	parser.add_argument('--dataset_file',
		metavar='',
		type=str,
		help='path to load the dataset',
		default='../dataset/dataset_test_bn.json')
	parser.add_argument('--model_name',
		metavar='',
		type=str,
		help='model to pretrain',
		default='bert_mini')
	parser.add_argument('--local_rank',
		metavar='',
		type=int,
		help='rank of the process during distributed training',
		default=-1)

	args = parser.parse_args()

	if not args.model_hash:
		test(args)
	else:
		main(args)

