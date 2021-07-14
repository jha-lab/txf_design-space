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
import json

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular

from boshnas import BOSHNAS
from aqn import gosh_aqn as aqn

from library import GraphLib, Graph
from utils import print_util as pu


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

CONF_INTERVAL = 0.005 # Corresponds to 0.5% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.9 # Corresponds to the minimum overlap for model to be considered

DEBUG = False
ACCURACY_PATIENCE = 10 # Convergence criterion for accuracy
ALEATORIC_QUERIES = 10 # Number of queries to be run with aleatoric uncertainty
K = 10 # Number of parallel cold restarts for BOSHNAS
UNC_PROB = 0.1
DIV_PROB = 0.1


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
		print(f'Model (with hash: {model_hash}) copied from neighbor. Finetuning model.')
	else:
		if model_hash not in os.listdir(f'{models_dir}pretrained/'):
			print(f'Model (with hash: {model_hash}) is not pretrained. Pretraining first.')
			pretrain = True
		else:
			print(f'Model (with hash: {model_hash}) is pretrained. Directly finetuning.')
		
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


def update_dataset(graphLib: 'GraphLib', finetune_dir: str):
	"""Update the dataset with all finetuned models
	
	Args:
	    graphLib (GraphLib): GraphLib opject to update
	    finetune_dir (str): directory with all finetuned models
	"""
	count = 0
	best_accuracy = 0
	for model_hash in os.listdir(finetune_dir):
		results_json = os.path.join(finetune_dir, model_hash, 'all_results.json')
		if os.path.exists(results_json):
			with open(results_json) as json_file:
				results = json.load(json_file)
				_, model_idx = graphLib.get_graph(model_hash=model_hash)
				graphLib.library[model_idx].accuracy = results['eval_accuracy']
				if results['eval_accuracy'] > best_accuracy:
					best_accuracy = results['eval_accuracy']
				count += 1

	print(f'{pu.bcolors.OKGREEN}Trained points in dataset:{pu.bcolors.ENDC} {count}' \
		+ f'{pu.bcolors.OKGREEN}Best accuracy:{pu.bcolors.ENDC} {best_accuracy}')

	return best_accuracy


def convert_to_tabular(graphLib: 'GraphLib'):
	"""Convert the graphLib object to a tabular dataset from 
	input encodings to the output loss
	
	Args:
	    graphLib (GraphLib): GraphLib object
	
	Returns:
	    X, y (tuple): input embeddings and output loss
	"""
	X, y = [], []
	for graph in graphLib.library:
		if graph.accuracy:
			X.append(graph.embedding)
			y.append(1 - graph.accuracy)

	X, y = np.array(X), np.array(y)

	return X, y


def get_neighbor_hash(model: 'Graph', trained_hashes: list):
	# Create BertConfig for the current model
	model_config = BertConfig()
    model_config.from_model_dict(model.model_dict)

    chosen_neighbor_hash = None
    max_overlap = 0

    # Choose neighbor with max overlap given that it adhere to the overlap constraint
	for neighbor_hash in model.neighbors:
		if neighbor_hash not in trained_hashes: 
			# Neighbor should be trained for weight transfer
			continue

		# Initialize current model
        current_model = BertModelModular(model_config)

        # Initialize neighbor model
        neighbor_graph, _ = graphLib.get_graph(model_hash=neighor_hash)
        neighbor_config = BertConfig(neighbor_graph.model_dict)
        neighbor_config.from_model_dict(neighbor_graph.model_dict)
        neighbor_model = BertModelModular(neighbor_config)

        # Get overlap from neighboring model
        overlap = current_model.load_model_from_source(neighbor_model)

        if overlap >= OVERLAP_THRESHOLD:
            train_model = True
            if overlap >= max_overlap:
                max_overlap = overlap
                chosen_neighbor_hash = neighbor_hash

    return chosen_neighbor_hash


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

	# New dataset file
    new_dataset_file = args.dataset_file.split('.json')[0] + f'_{args.task}.json'

	# 1. Pre-train randomly sampled models if number of pretrained models available 
	#   is less than num_init
	# 2. Fine-tune pretrained models for the given task
	# 3. Train BOSHNAS on initial models
	# 4. With prob 1 - epsilon - delta, train models with best acquisition function
	#   - add only those models to queue that have high overlap and finetune
	#   - if no model with high overlap neighbors, pretrain best predicted model(s)
	# 5. With prob epsilon: train models with max std, and prob delta: train random models
	#   - if neighbor pretrained with high overlap, finetune only; else pretrain
	# 6. Keep a dictionary of job id and model hash. Get accuracy from trained model using 
	#   the model hash and all_results.json. Wait for spawning more jobs
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
			epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, 
			cluster=args.cluster, id=args.id)
		assert pretrain is False
		
		model_jobs.append({'model_hash': model_hash, 
			'job_id': job_id, 
			'train_type': 'P+F' if pretrain else 'F'})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Pretrain and then finetune randomly sampled models if total finetuned models are less than num_init
	# TODO: Add skopt.sampler.Sobol points instead
	while len(pretrained_hashes) < args.num_init:
		sample_idx = random.randint(0, len(graphLib))
		model_hash = graphLib.library[sample_idx].hash

		if model_hash not in pretrained_hashes:
			pretrained_hashes.append(model_hash)

			model, model_idx = graphLib.get_graph(model_hash=model_hash)

			job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
				epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, 
				cluster=args.cluster, id=args.id)
			assert pretrain is True
			
			model_jobs.append({'model_hash': model_hash, 
				'job_id': job_id, 
				'train_type': 'P+F' if pretrain else 'F'})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Update dataset with newly trained models
	old_best_accuracy = update_dataset(graphLib)

	# Get entire dataset in embedding space
	X_ds = []
	for graph in graphLib.library:
		X_ds.append(graph.embedding)
	X_ds = np.array(X_ds)

	min_X, max_X = np.min(X_ds, axis=0), np.max(X_ds, axis=0)

	# Initialize the BOSHNAS model
	surrogate_model = BOSHNAS(input_dim=X_ds.shape[1],
							  bounds=(min_X, max_X),
							  trust_region=False,
							  second_order=True,
							  parallel=True if not DEBUG else False,
							  model_aleatoric=True,
							  save_path=args.surrogate_model_dir,
							  pretrained=False)

	# Get initial dataset after finetuning num_init models
	X, y = convert_to_tabular(graphLib)
	max_loss = np.amax(y)

	same_accuracy = 0
	method = 'optimization'

	while same_accuracy < ACCURACY_PATIENCE + ALEATORIC_QUERIES:
		prob = random.uniform(0, 1)
		if 0 <= prob <= (1 - UNC_PROB - DIV_PROB):
			method = 'optimization'
		elif 0 <= prob <= (1 - DIV_PROB):
			method = 'unc_sampling'
		else:
			method = 'div_sampling'

		# Get a set of trained models and models that are currently in the pipeline
		trained_hashes, pipeline_hashes = [], []
		for job in model_jobs:
			_, _, status = get_job_info(job['job_id'])
			if status == 'COMPLETED':
				trained_hashes.append(job['model_hash'])
			else:
				pipeline_hashes.append(job['model_hash'])

		new_queries = 0

		if method == 'optimization'
			print(f'{pu.bcolors.OKBLUE}Running optimization step{pu.bcolors.ENDC}')
			# Get current tabular dataset
			X, y = convert_to_tabular(graphLib)
			y = y/max_loss

			# Train BOSHNAS model
			train_error = surrogate_model.train(X, y)

			# Use aleatoric loss close to convergence to optimize training recipe
			if same_accuracy < ACCURACY_PATIENCE:
				# Architecture not converged yet. Use only epistemic uncertainty
				use_al = False
			else:
				# Use aleatoric uncertainty to optimize training recipe
				use_al = True

			# Get next queries
			query_indices = surrogate_model.get_queries(x=X_ds, k=K, explore_type='ucb', use_al=use_al)

			# Run queries
			for i in set(query_indices):
				model = graphLib.library[i]

				if not use_al and model.hash in trained_hashes + pipeline_hashes:
					# If aleatoric uncertainty is not considered, only consider models that are not 
					# already trained or in the pipeline
					continue

				chosen_neighbor_hash = get_neighbor_hash(model, trained_hashes)

	            if chosen_neighbor_hash:
	            	# Finetune model with the chosen neighbor
					job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
						epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=chosen_neighbor_hash, 
						cluster=args.cluster, id=args.id)
					assert pretrain is False
				else:
					# If no neighbor was found with high overlap, proceed to next query model
					continue

				new_queries += 1
				
				model_jobs.append({'model_hash': model_hash, 
					'job_id': job_id, 
					'train_type': 'P+F' if pretrain else 'F'})
	  		
	  		if new_queries == 0:
	  			# If no queries were found where direct finetuning could be performed, pretrain model
	  			# with best acquisition function value
	  			query_embeddings = [X_ds[idx, :] for idx in query_indices]
	  			candidate_predictions = surrogate_model.predict(query_embeddings)

	  			best_prediction_index = query_indices[np.argmax(aqn([pred[0] for pred in candidate_predictions],
	  															[pred[1][0] for pred in candidate_predictions],
	  															explore_type='ucb'))]

	  			# Pretrain model
				job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
					epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, 
					cluster=args.cluster, id=args.id)
				assert pretrain is True

		elif method == 'unc_sampling':
			print(f'{pu.bcolors.OKBLUE}Running uncertainty sampling{pu.bcolors.ENDC}')

			candidate_predictions = surrogate_model.predict(X_ds)

			# Get model index with highest epistemic uncertainty
			unc_prediction_idx = np.argmax([pred[1][0] for pred in candidate_predictions])

			# Sanity check: model with highest epistemic uncertainty should not be trained
			assert graphLib.library[unc_prediction_idx].hash not in trained_hashes

			if graphLib.library[unc_prediction_idx].hash in pipeline_hashes:
				print(f'{pu.bcolors.OKBLUE}Highest uncertainty model already in pipeline{pu.bcolors.ENDC}')
			else:
				model = graphLib.library[unc_prediction_idx]

				# Pretrain sampled architecture
				job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
						epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, 
						cluster=args.cluster, id=args.id)
				assert pretrain is True

				new_queries += 1
				
				model_jobs.append({'model_hash': model_hash, 
					'job_id': job_id, 
					'train_type': 'P+F' if pretrain else 'F'})

		else:
			print(f'{pu.bcolors.OKBLUE}Running diversity sampling{pu.bcolors.ENDC}')

			# Get randomly sampled model idx
			# TODO: Add skopt.sampler.Sobol points instead
			unc_prediction_idx = random.randint(0, len(graphLib))

			while graphLib.library[unc_prediction_idx].hash in trained_hashes + pipeline_hashes:
				unc_prediction_idx = random.randint(0, len(graphLib))

			model = graphLib.library[unc_prediction_idx]

			# Pretrain sampled architecture
			job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
					epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, 
					cluster=args.cluster, id=args.id)
			assert pretrain is True

			new_queries += 1
			
			model_jobs.append({'model_hash': model_hash, 
				'job_id': job_id, 
				'train_type': 'P+F' if pretrain else 'F'})

		# Wait for jobs to complete
		wait_for_jobs(model_jobs)

		# Update dataset with newly trained models
		best_accuracy = update_dataset(graphLib)

		# Update same_accuracy to check convergence
		if best_accuracy == old_best_accuracy and method == 'optimization':
			same_accuracy += 1

		old_best_accuracy = best_accuracy

	# Wait for jobs to complete
	wait_for_jobs(model_jobs, running_limit=0, patience=0)

	# Update dataset with newly trained models
	best_accuracy = update_dataset(graphLib)

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()

