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


def get_training_args(models_dir, task, id, model_hash, autotune, autotune_trials):

	model_name_or_path = f'{models_dir}pretrained/{model_hash}'

	training_args = f'--model_name_or_path {model_name_or_path} \
		--task_name {task} \
		--id {id} \
		--do_train \
		--do_eval \
		{"--autotune" if autotune else ""} \
		--autotune_trials {autotune_trials} \
		--logging_steps 50 \
		--max_seq_length 512 \
		--per_device_train_batch_size 64 \
		--load_best_model_at_end \
		--metric_for_best_model eval_loss \
		--learning_rate 2e-5 \
		--weight_decay 0.01 \
		--num_train_epochs 5 \
		--overwrite_output_dir \
		--fp16 \
		--output_dir {models_dir}{task}/{model_hash}/'

	training_args = shlex.split(training_args)

	return training_args


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
	parser.add_argument('--id',
		metavar='',
		type=str,
		help='PU Net ID',
		default='stuli')
	parser.add_argument('--autotune',
		dest='autotune',
		action='store_true')
	parser.add_argument('--autotune_trials',
		metavar='',
		type=int,
		help='number of trials for optuna',
		default=20)
	parser.set_defaults(autotune=False)

	args = parser.parse_args()
	
	glue_scores = {}
	score = 0

	for task in GLUE_TASKS:

		autotune = args.autotune # and not (task=='qqp' or task == 'qnli')
		training_args = get_training_args(args.models_dir, task, args.id, args.model_hash, autotune, args.autotune_trials)
		metrics = finetune(training_args)

		if task == 'cola':

			glue_scores[task] = metrics['eval_matthews_correlation']
			task_score = glue_scores[task]

		elif task == 'stsb':

			glue_scores[task+'_spearman'] = metrics['eval_spearmanr']
			glue_scores[task+'_pearson'] = metrics['eval_pearson']
			task_score = max(metrics['eval_spearmanr'], metrics['eval_pearson']) # (metrics['eval_spearmanr']+metrics['eval_pearson'])/2.0

		elif task == 'mrpc' or task == 'qqp':

			glue_scores[task+'_accuracy'] = metrics['eval_accuracy']
			glue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = max(metrics['eval_accuracy'], metrics['eval_f1']) # (metrics['eval_accuracy']+metrics['eval_f1'])/2.0

		elif task in ["sst2", "mnli",  "qnli", "rte", "wnli"]:

			glue_scores[task] = metrics['eval_accuracy']
			task_score = metrics['eval_accuracy']
			
		#print(task,':',task_score)
				
		score+=task_score

	glue_scores['glue_score'] = score*1.0/9.0
						
	print(f"GLUE score for model with hash '{args.model_hash}': {score*1.0/9.0}")

	output_dir = f"{args.models_dir}glue/{args.model_hash}/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_dir+'all_results.json', 'w') as fp:
		json.dump(glue_scores, fp)

if __name__ == '__main__':
 
	main()
