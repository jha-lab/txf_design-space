import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import logging
# logging.disable(logging.INFO)
# logging.disable(logging.WARNING)

import argparse

import pdb

from finetune_flexibert import finetune
import shlex

from utils import print_util as pu

import json


SUPERGLUE_TASKS = ['boolq', 'cb', 'copa', 'multirc', 'wic', 'wsc.fixed']

SUPERGLUE_TASKS_DATASET = ['BoolQ', 'CB', 'COPA', 'MultiRC', 'WiC', 'WSC']


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
		--gradient_accumulation_steps 1 \
		--load_best_model_at_end \
		--metric_for_best_model eval_loss \
		--learning_rate 2e-5 \
		--weight_decay 0.01 \
		--num_train_epochs 5 \
		--overwrite_output_dir \
		--output_dir {models_dir}/superglue/{model_hash}/{task}/'

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

	os.makedirs(f'{args.models_dir}/superglue/{args.model_hash}', exist_ok=True)
	
	superglue_scores = {}
	score = 0

	for task in SUPERGLUE_TASKS:

		print(f'Running task: {task}')
		autotune = args.autotune
		training_args = get_training_args(args.models_dir, task, args.id, args.model_hash, autotune, args.autotune_trials)
		metrics = finetune(training_args)

		if task == 'axb':
			superglue_scores[task] = metrics['eval_matthews_correlation']
			task_score = superglue_scores[task]

		elif task in ['multirc', 'record', 'wsc.fixed']:
			superglue_scores[task+'_accuracy'] = metrics['eval_accuracy']
			superglue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = max(metrics['eval_accuracy'], metrics['eval_f1']) # (metrics['eval_accuracy']+metrics['eval_f1'])/2.0

		elif task == 'cb':
			superglue_scores[task+'_f1'] = metrics['eval_f1']
			task_score = metrics['eval_f1']

		else:
			superglue_scores[task] = metrics['eval_accuracy']
			task_score = metrics['eval_accuracy']
				
		score += task_score

	superglue_scores['glue_score'] = score / 6.0
						
	print(f"SuperGLUE score for model with hash '{args.model_hash}': {score / 6.0}")

	output_dir = f"{args.models_dir}superglue/{args.model_hash}/"

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_dir+'all_results.json', 'w') as fp:
		json.dump(superglue_scores, fp)

if __name__ == '__main__':
 
	main()
