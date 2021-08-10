import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

import argparse

import pdb

from finetune_flexibert import finetune
import shlex

from library import GraphLib, Graph
from utils import print_util as pu

import json


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

GLUE_TASKS_DATASET = ['CoLA', 'MNLI-mm', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']


def get_training_args(models_dir, task, model_hash, autotune, autotune_trials):

	model_name_or_path = f'{models_dir}pretrained/{model_hash}'

	training_args = f'--model_name_or_path {model_name_or_path} \
		--task_name {task} \
		--do_train \
		--do_eval \
		{"--autotune" if autotune else ""} \
		--autotune_trials {autotune_trials} \
		--logging_steps 50 \
		--max_seq_length 128 \
		--per_device_train_batch_size 64 \
		--load_best_model_at_end \
		--learning_rate 2e-5 \
		--num_train_epochs 4 \
		--overwrite_output_dir \
		--output_dir {models_dir}{task}/{model_hash}/'

	training_args = shlex.split(training_args)

	return training_args

def test():

	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset_file',
		metavar='',
		type=str,
		help='path to load the dataset',
		default='../dataset/dataset_test.json')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to "models" directory containing "pretrained" sub-directory',
		default='../models/')
	parser.add_argument('--model_name',
		metavar='',
		type=str,
		help='model name'
		)

	args = parser.parse_args()

	if args.model_name == 'bert_mini':

		model_dict = {'l': 4, 'o': ['sa']*4, 'h': [256]*4, 'n': [4]*4, 'f': [[1024]]*4, 'p': ['sdp']*4}


	elif args.model_name == 'bert_tiny':

		model_dict = { 'l': 2, 'o': ['sa']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['sdp']*2}


	elif args.model_name == 'fnet_mini':
		
		model_dict = {'l':4,'o':['l','l','l','l'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':['dft','dft','dft','dft']}

	elif args.model_name == 'fnet_tiny':

		model_dict =  { 'l': 2, 'o': ['l']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['dft']*2}

	elif args.model_name == 'convbert_mini':

		model_dict = {'l':4,'o':['c','c','c','c'],'h':[256,256,256,256],'n':[4,4,4,4],'f':[[1024],[1024],[1024],[1024]],'p':[9,9,9,9]}

	elif args.model_name == 'convbert_tiny':

		model_dict= { 'l': 2, 'o': ['c']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': [9]*2}

	elif args.model_name == 'bert_2_256':

		model_dict = { 'l': 2, 'o': ['sa']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['sdp']*2}

	elif args.model_name == 'bert_4_128':

		model_dict = { 'l': 4, 'o': ['sa']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['sdp']*4}


	elif args.model_name == 'fnet_2_256':

		model_dict = { 'l': 2, 'o': ['l']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['dft']*2}

	elif args.model_name == 'fnet_4_128':

		model_dict = { 'l': 4, 'o': ['l']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['dft']*4}


	elif args.model_name == 'convbert_2_256':

		model_dict = { 'l': 2, 'o': ['c']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': [9]*2}

	elif args.model_name == 'convbert_4_128':

		model_dict = { 'l': 4, 'o': ['c']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': [9]*4}


	graphLib = GraphLib.load_from_dataset(args.dataset_file)
	graph = graphLib.get_graph(model_dict=model_dict)[0]
	
	
	glue_scores = {}
	score = 0

	for task in GLUE_TASKS:

		training_args = get_training_args(args.models_dir, task, graph.hash)
		metrics = finetune(training_args)

		if task == 'cola':

			glue_scores[task] = metrics['eval_matthews_correlation']
			task_score = glue_scores[task]

		elif task == 'stsb':

			glue_scores[task+'_spearman'] = metrics['eval_spearmanr']
			glue_scores[task+'_pearson'] = metrics['eval_pearson']
			task_score = (metrics['eval_spearmanr']+metrics['eval_pearson'])/2

		elif task == 'mrpc' or task=='qqp':

			glue_scores[task+'_accuracy'] = metrics['eval_accuracy']
			glue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = (metrics['eval_accuracy']+metrics['eval_f1'])/2

		elif task in ["sst2", "mnli",  "qnli", "rte", "wnli"]:

			glue_scores[task] = metrics['eval_accuracy']
			task_score = metrics['eval_accuracy']
			
		print(task,':',task_score)
				
		score+=task_score
						
	
	print(f"{model_name}:", score*1.0/9)

	output_dir = f"../models/glue_score/{model_hash}/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_dir+'glue_score.json', 'w') as fp:
		json.dump(glue_scores, fp)

	for key, value in glue_scores:

		score+=value
		total +=1
		print(key,':',value)

	print('SCORE:', score*1.0/total)

	output_dir = f"{models_dir}glue_scores/{graph.hash}.json"

	with open(output_dir, 'w') as fp:
		json.dump(glue_scores, fp)


def main():
	parser = argparse.ArgumentParser(
		description='Input parameters for glue score computation',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to "models" directory containing "pretrained" sub-directory',
		default='../models/')
	parser.add_argument('--model_hash',
		metavar='',
		type=str,
		help='hash of the given model')
	parser.add_argument('--autotune',
		dest='autotune',
		action='store_true')
	parser.add_argument('--autotune_trials',
		metavar='',
		type=int,
		help='number of trials for optuna',
		default=5)
	parser.set_defaults(autotune=False)

	args = parser.parse_args()
	
	glue_scores = {}
	score = 0

	for task in GLUE_TASKS:

		autotune = args.autotune and not( task=='qqp' or task == 'qnli')
		training_args = get_training_args(args.models_dir, task, args.model_hash, autotune, args.autotune_trials)
		metrics = finetune(training_args)

		if task == 'cola':

			glue_scores[task] = metrics['eval_matthews_correlation']
			task_score = glue_scores[task]

		elif task == 'stsb':

			glue_scores[task+'_spearman'] = metrics['eval_spearmanr']
			glue_scores[task+'_pearson'] = metrics['eval_pearson']
			task_score = (metrics['eval_spearmanr']+metrics['eval_pearson'])/2

		elif task == 'mrpc' or task == 'qqp':

			glue_scores[task+'_accuracy'] = metrics['eval_accuracy']
			glue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = (metrics['eval_accuracy']+metrics['eval_f1'])/2

		elif task in ["sst2", "mnli",  "qnli", "rte", "wnli"]:

			glue_scores[task] = metrics['eval_accuracy']
			task_score = metrics['eval_accuracy']
			
		#print(task,':',task_score)
				
		score+=task_score

	glue_scores['glue_score'] = score*1.0/9
						
	# print(f"{args.model_hash}:", score*1.0/9)

	output_dir = f"{args.models_dir}glue/{args.model_hash}/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_dir+'all_results.json', 'w') as fp:
		json.dump(glue_scores, fp)

if __name__ == '__main__':
 
	main()











