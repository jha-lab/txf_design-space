# Run Bayesian Optimization using Second-order gradients and a Heteroscedastic
# surrogate model for Network Architecture Search (BOSHNAS) on all Transformers
# in FlexiBERT's design space.

# Author : Shikhar Tuli


import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')
sys.path.append('../boshnas/')

import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

import argparse
import numpy as np
import random
import tabulate
import subprocess
import time

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular

from boshnas import BOSHNAS

from library import GraphLib, Graph
from utils import print_util as pu


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

CONF_INTERVAL = 0.005 # Corresponds to 0.5% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.9 # Corresponds to the minimum overlap for model to be considered

DEBUG = False


def worker(model_dict: dict,
	model_hash: str,
	task: str, 
	epochs: int, 
	models_dir: str,
	chosen_neighbor_hash: str,
	cluster: str,
	id: str):
	"""Worker to finetune or pretrain the given model
	
	Args:
		model_idx (int): index for the model in shared_accuracies
		model_dict (dict): model dictionary
		model_hash (str): hash of the given model
		task (str): name of the GLUE task for fine-tuning the model on; should
			be in GLUE_TASKS
		epochs (int): number of epochs for fine-tuning
		models_dir (str): path to "models" directory containing "pretrained" sub-directory
		chosen_neighbor_hash (str, optional): hash of the chosen neighbor
		cluster (str): name of the cluster - "adroit" or "tiger"
		id (str): PU-NetID that is used to run slurm commands
	
	Returns:
		job_id, pretrain (str, bool): Job ID for the slurm scheduler and whether pretraining
		is being performed
	"""
	pretrain = False

	if chosen_neighbor_hash is not None:
		# Load weights of current model using the finetuned neighbor that was chosen
		model_config = BertConfig()
		model_config.from_model_dict(model_dict)
		chosen_neighbor_model = BertModelModular.from_pretrained(
			f'{models_dir}{task}/{chosen_neighbor_hash}/')
		current_model = BertModelModular(model_config)
		current_model.load_model_from_source(chosen_neighbor_model)
		current_model.save_pretrained(
			f'{models_dir}{task}/{model_hash}/')

		model_name_or_path = f'{models_dir}{task}/{model_hash}/' 
		print(f'Model (with hash: {model_hash}) copied from neighbor. Fine-tuning model.')
	else:
		if model_hash not in os.listdir(f'{models_dir}pretrained/'):
			print(f'Model (with hash: {model_hash}) is not pretrained. Pre-training first.')
			pretrain = True
		else:
			print(f'Model (with hash: {model_hash}) is pretrained. Directly fine-tuning.')
		
		model_name_or_path = f'{models_dir}pretrained/{model_hash}/'

	args = ['--task', task]
	args.extend(['--cluster', cluster])
	args.extend(['--id', id])
	args.extend(['--pretrain', '1' if pretrain else '0'])
	args.extend(['--model_hash', model_hash])
	args.extend(['--model_name_or_path', model_name_or_path])
	args.extend(['--epochs', str(epochs)])
	args.extend(['--output_dir', f'{models_dir}{task}/{model_hash}/'])

	slurm_stdout = subprocess.check_output(f'source ./job_scripts/job_train_script.sh {" ".join(args)}',
		shell=True, text=True)

	return slurm_stdout.split()[-1], pretrain
		

def get_job_info(job_id: int):
	"""Obtain job info
	
	Args:
		job_id (int): job id
	
	Returns:
		start_time, elapsed_time, status (str, str, str): job details
	"""
	slurm_stdout = subprocess.check_output(f'slist {job_id}', shell=True, text=True)
	slurm_stdout = slurm_stdout.split('\n')[2].split()

	if len(slurm_stdout) > 7:
		start_time, elapsed_time, status = slurm_stdout[5], slurm_stdout[6], slurm_stdout[7]
		if start_time == 'Unknown': start_time = 'UNKNOWN'
	else:
		start_time, elapsed_time, status = 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

	return start_time, elapsed_time, status


def print_jobs(model_jobs: list):
	"""Print summary of all completed, pending and running jobs
	
	Args:
		model_jobs (list): list of jobs
	"""
	header = ['MODEL HASH', 'JOB ID', 'TRAIN TYPE', 'START TIME', 'ELAPSED TIME', 'STATUS']

	rows = []
	for job in model_jobs:
		start_time, elapsed_time, status = get_job_info(job['job_id'])
		rows.append([job['model_hash'], job['job_id'], job['train_type'], start_time, elapsed_time, status])

	print()
	print(tabulate.tabulate(rows, header))


def wait_for_jobs(model_jobs: list, running_limit: int = 4, patience: int = 1):
	"""Wait for current jobs in queue to complete
	
	Args:
		model_jobs (list): list of jobs
		running_limit (int, optional): nuber of running jobs to limit
		patience (int, optional): number of pending jobs to wait for
	"""
	print_jobs(model_jobs)

	completed_jobs = 0
	last_completed_jobs = 0
	running_jobs = np.inf
	pending_jobs = np.inf
	while running_jobs >= running_limit or pending_jobs > patience:
		completed_jobs, running_jobs, pending_jobs = 0, 0, 0
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED': 
				completed_jobs += 1
			elif status == 'PENDING':
				pending_jobs += 1
			elif status == 'RUNNING':
				running_jobs += 1
		if last_completed_jobs != completed_jobs:
			print_jobs(model_jobs)
		last_completed_jobs = completed_jobs 
		time.sleep(1)


def main():
	"""Run BOSHNAS to get the best architecture in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset_file',
		metavar='',
		type=str,
		help='path to load the dataset',
		default='../dataset/dataset_small.json')
	parser.add_argument('--task',
		metavar='',
		type=str,
		help=f'name of GLUE tasks to train surrogate model for',
		default='sst2')
	parser.add_argument('--epochs',
		metavar='',
		type=int,
		help=f'number of epochs to finetune',
		default=5)
	parser.add_argument('--surrogate_model_dir',
		metavar='',
		type=str,
		help='path to save the surrogate model parameters',
		default='../dataset/surrogate_models/sst2/')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to "models" directory containing "pretrained" sub-directory',
		default='../models/')
	parser.add_argument('--num_init',
		metavar='',
		type=int,
		help='number of initial models to initialize the BOSHNAS model',
		default=10)
	parser.add_argument('--n_jobs',
		metavar='',
		type=int,
		help='number of parallel jobs for training BOSHNAS',
		default=8)
	parser.add_argument('--cluster',
		metavar='',
		type=str,
		help='name of the cluster - "adroit" or "tiger"',
		default='della')
	parser.add_argument('--id',
		metavar='',
		type=str,
		help='PU-NetID that is used to run slurm commands',
		default='stuli')

	args = parser.parse_args()

	random_seed = 1

	assert args.task in GLUE_TASKS, f'GLUE task should be in: {GLUE_TASKS}'

	# Take global dataset and assign task
	graphLib = GraphLib.load_from_dataset(args.dataset_file)
	graphLib.dataset = args.task

	# 1. Pre-train randmly sampled models if number of pretrained models available 
	#   is less than num_init
	# 2. Fine-tune pretrained models for the given task
	# 3. Train BOSHNAS on initial models
	# 4. With prob 1 - epsilon - delta, train models with best acquisition function
	#   - add only those models to queue that have high overlap and finetune
	#   - if no model with high overlap neighbors, pretrain best predicted model(s)
	# 5. With prob epsilon: train models with max std, and prob delta: train random models
	#   - if neighbor pretrained with high overlap, finetune only; else pretrain
	# 6. Keep a dictionary of job id and model hash. Get accuracy from trained model using 
	#   the model hash and all_results.json. Wait for spawning more jobs if some jobs are 
	#   waiting using: for job in 
	#   subprocess.check_output(f'squeue -u {args.id}', shell=True, text=True).split('\n'), 
	#   check if 'PD' in job.split(). Use a patience factor
	# 7. Update the BOSHNAS model and put next queries in queue
	# 8. Optional: Use aleatoric uncertainty and re-finetune models if accuracy converges
	# 9. Stop training if a stopping criterion is reached
	
	# Initialize a dictionary mapping the model hash to its corresponding job_id
	model_jobs = []

	pretrain_dir = os.path.join(args.models_dir, 'pretrained')
	finetune_dir = os.path.join(args.models_dir, args.task)

	pretrained_hashes = os.listdir(pretrain_dir)
	# pretrained_hashes = random.sample([graph.hash for graph in graphLib.library], 4)

	# Finetune pretrained models
	for model_hash in pretrained_hashes:
		model, model_idx = graphLib.get_graph(model_hash=model_hash)
		
		job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
			epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, cluster=args.cluster, id=args.id)
		assert pretrain is False
		
		model_jobs.append({'model_hash': model_hash, 
			'job_id': job_id, 
			'train_type': 'P+F' if pretrain else 'F'})


	wait_for_jobs(model_jobs)

	# Pretrain and then finetune randomly sampled models if total finetuned models are less than num_init
	while len(pretrained_hashes) < args.num_init:
		sample_idx = random.randint(0, len(graphLib))
		model_hash = graphLib.library[sample_idx].hash

		if model_hash not in pretrained_hashes:
			pretrained_hashes.append(model_hash)

			model, model_idx = graphLib.get_graph(model_hash=model_hash)

			job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
				epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, cluster=args.cluster, id=args.id)
			assert pretrain is True
			
			model_jobs.append({'model_hash': model_hash, 
				'job_id': job_id, 
				'train_type': 'P+F' if pretrain else 'F'})

	wait_for_jobs(model_jobs)

	raise NotImplementedError('BOSHNAS has not been implemented yet')    


if __name__ == '__main__':
	main()

