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
import copy
import shutil

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular

from boshnas import BOSHNAS
from acq import gosh_acq as acq

from library import GraphLib, Graph
from utils import print_util as pu


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

CONF_INTERVAL = 0.005 # Corresponds to 0.5% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.8 # Corresponds to the minimum overlap for model to be considered

DEBUG = False
ACCURACY_PATIENCE = 5 # Convergence criterion for accuracy
ALEATORIC_QUERIES = 5 # Number of queries to be run with aleatoric uncertainty
K = 10 # Number of parallel cold restarts for BOSHNAS
UNC_PROB = 0.1
DIV_PROB = 0.1
HOMOGENEOUS_ONLY = False
if HOMOGENEOUS_ONLY:
	UNC_PROB, DIV_PROB = 0, 1


def worker(model_dict: dict,
	model_hash: str,
	task: str, 
	epochs: int, 
	models_dir: str,
	chosen_neighbor_hash: str,
	dataset_file: str,
	autotune: bool,
	autotune_trials: int):
	"""Worker to finetune or pretrain the given model
	
	Args:
		model_dict (dict): model dictionary
		model_hash (str): hash of the given model
		task (str): name of the GLUE task for fine-tuning the model on; should
			be in GLUE_TASKS, or "glue"
		epochs (int): number of epochs for fine-tuning
		models_dir (str): path to "models" directory containing "pretrained" sub-directory
		chosen_neighbor_hash (str): hash of the chosen neighbor
		dataset_file (str): path to the dataset file
		autotune (bool): whether to automatically tune the training recipe
		autotune_trials (int): number of trials for autotuning. Only used if autotune is True
	
	Returns:
		job_id, pretrain (str, bool): Job ID for the slurm scheduler and whether pretraining
		is being performed
	"""
	pretrain = False

	if chosen_neighbor_hash is not None:
		# Load weights of current model using the finetuned neighbor that was chosen
		model_config = BertConfig()
		model_config.from_model_dict(model_dict)

		if task != "glue":
			chosen_neighbor_model = BertModelModular.from_pretrained(
				f'{models_dir}{task}/{chosen_neighbor_hash}/')
			current_model = BertModelModular(model_config)
			current_model.load_model_from_source(chosen_neighbor_model)
			current_model.save_pretrained(
				f'{models_dir}{task}/{model_hash}/')
			model_name_or_path = f'{models_dir}{task}/{model_hash}/' 
		else:
			for task in GLUE_TASKS:
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
	args.extend(['--partition', partition])
	args.extend(['--dataset_file', dataset_file])
	args.extend(['--pretrain', '1' if pretrain else '0'])
	args.extend(['--autotune', '1' if autotune else '0'])
	args.extend(['--autotune_trials', str(autotune_trials)])
	args.extend(['--model_hash', model_hash])
	args.extend(['--model_name_or_path', model_name_or_path])
	args.extend(['--models_dir', models_dir])
	args.extend(['--epochs', str(epochs)])
	args.extend(['--output_dir', f'{models_dir}{task}/{model_hash}/'])

	command = f'source ./job_scripts/job_train_script.sh {" ".join(args)}'

	slurm_stdout = subprocess.check_output(command, shell=True, text=True)

	return slurm_stdout.split()[-1], pretrain


def get_job_info(job_id: int):
	"""Obtain job info
	
	Args:
		job_id (int): job id
	
	Returns:
		start_time, elapsed_time, status (str, str, str): job details
	"""
	slurm_stdout = subprocess.check_output(f'ssh della-gpu "slist {job_id}"', shell=True, text=True)
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

	last_completed_jobs = 0
	last_running_jobs = 0
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
			elif status == 'FAILED':
				print_jobs(model_jobs)
				raise RuntimeError('Some jobs failed.')

		if last_completed_jobs != completed_jobs or last_running_jobs != running_jobs:
			print_jobs(model_jobs)

		last_completed_jobs, last_running_jobs = completed_jobs, running_jobs 


def update_dataset(graphLib: 'GraphLib', task: str, finetune_dir: str, dataset_file: str):
	"""Update the dataset with all finetuned models
	
	Args:
		graphLib (GraphLib): GraphLib opject to update
		task (str): name of GLUE task (or "glue")
		finetune_dir (str): directory with all finetuned models
		dataset_file (str): path the the dataset file where updated graphLib is stored
	"""
	count = 0
	best_performance = 0
	best_hash = ''
	
	for model_hash in os.listdir(finetune_dir):
		metrics_json = os.path.join(finetune_dir, model_hash, 'all_results.json')
		if os.path.exists(metrics_json):
			with open(metrics_json) as json_file:
				metrics = json.load(json_file)
				_, model_idx = graphLib.get_graph(model_hash=model_hash)

				if task == 'cola':
					performance = metrics['eval_matthews_correlation']
				elif task == 'stsb':
					performance = (metrics['eval_spearmanr'] + metrics['eval_pearson']) / 2
				elif task in ['mrpc', 'qqp']:
					performance = (metrics['eval_accuracy'] + metrics['eval_f1']) / 2
				elif task in ['sst2', 'mnli',  'qnli', 'rte', 'wnli']:
					performance = metrics['eval_accuracy']
				elif task == 'glue':
					performance = metrics['glue_score']
				else:
					raise ValueError(f'The given task: {task} is not supported')

				graphLib.library[model_idx].performance = performance

				if performance > best_performance:
					best_performance = performance
					best_hash = model_hash
				count += 1

	print()
	graphLib.save_dataset(dataset_file)

	print(f'\n{pu.bcolors.OKGREEN}Trained points in dataset:{pu.bcolors.ENDC} {count}\n' \
		+ f'{pu.bcolors.OKGREEN}Best performance:{pu.bcolors.ENDC} {best_performance}\n' 
		+ f'{pu.bcolors.OKGREEN}Best model hash:{pu.bcolors.ENDC} {best_hash}\n')

	return best_performance


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
		if graph.performance:
			X.append(graph.embedding)
			y.append(1 - graph.performance)

	X, y = np.array(X), np.array(y)

	return X, y


def get_neighbor_hash(model: 'Graph', graphLib: 'GraphLib', trained_hashes: list):
	# Create BertConfig for the current model
	model_config = BertConfig()
	model_config.from_model_dict(model.model_dict)

	chosen_neighbor_hash = None
	max_overlap = 0

	# Choose neighbor with max overlap given where it follows the overlap constraint
	for neighbor_hash in model.neighbors:
		if neighbor_hash not in trained_hashes: 
			# Neighbor should be trained for weight transfer
			continue

		# Initialize current model
		current_model = BertModelModular(model_config)

		# Initialize neighbor model
		neighbor_graph, _ = graphLib.get_graph(model_hash=neighbor_hash)
		neighbor_config = BertConfig()
		neighbor_config.from_model_dict(neighbor_graph.model_dict)
		neighbor_model = BertModelModular(neighbor_config)

		# Get overlap from neighboring model
		overlap = current_model.load_model_from_source(neighbor_model)
		print(f'Overlap of query ({model.hash}) with neighbor ({neighbor_hash}) is: {overlap}')

		if overlap >= OVERLAP_THRESHOLD:
			if overlap >= max_overlap:
				max_overlap = overlap
				chosen_neighbor_hash = neighbor_hash

	return chosen_neighbor_hash


def is_homogenous(graphObject):
    model_dict = graphObject.model_dict
    hashed_f = [hash(str(item)) for item in model_dict['f']]
    return True if len(set(model_dict['h'])) == 1 and len(set(model_dict['n'])) == 1 and len(set(model_dict['o'])) == 1 \
        and len(set(hashed_f)) == 1 and len(set(model_dict['p'])) == 1 else False


def main():
	"""Run BOSHNAS to get the best architecture in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for running BOSHNAS over the FlexiBERT space',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset_file',
		metavar='',
		type=str,
		help='path to load the dataset',
		default='../dataset/dataset_test_bn.json')
	parser.add_argument('--task',
		metavar='',
		type=str,
		help=f'name of GLUE task (or "glue") to train surrogate model for',
		default='glue')
	parser.add_argument('--epochs',
		metavar='',
		type=int,
		help=f'number of epochs to finetune',
		default=5)
	parser.add_argument('--surrogate_model_dir',
		metavar='',
		type=str,
		help='path to save the surrogate model parameters',
		default='../dataset/surrogate_models/glue/')
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
	parser.add_argument('--autotune',
		dest='autotune',
		action='store_true')
	parser.add_argument('--autotune_trials',
		metavar='',
		type=int,
		help='number of trials for optuna',
		default=5)
	parser.add_argument('--n_jobs',
		metavar='',
		type=int,
		help='number of parallel jobs for training BOSHNAS',
		default=8)
	parser.set_defaults(autotune=False)

	args = parser.parse_args()
	autotune_trials = args.autotune_trials
	
	random_seed = 1

	assert args.task in GLUE_TASKS + ['glue'], f'given task should be in: {GLUE_TASKS} or "glue"'

	# Take global dataset and assign task
	graphLib = GraphLib.load_from_dataset(args.dataset_file)
	graphLib.dataset = args.task

	# New dataset file
	new_dataset_file = args.dataset_file.split('.json')[0] + f'_{args.task}.json'

	# Pseudo-code for the BOSHNAS pipeline:
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
	# 9. Stop training if a stopping criterion is met
	
	# Initialize a dictionary mapping the model hash to its corresponding job_id
	model_jobs = []

	pretrain_dir = os.path.join(args.models_dir, 'pretrained')
	finetune_dir = os.path.join(args.models_dir, args.task)

	# Load model jobs if previous instance of BOSHNAS ended unexpectedly
	model_jobs_file = f'./job_scripts/{args.task}/model_jobs.json'
	if os.path.exists(model_jobs_file):
		model_jobs = json.load(open(model_jobs_file))

		try:
			wait_for_jobs(model_jobs)
		except:
			# Is there are failed jobs, remove them
			model_jobs_copy = copy.deepcopy(model_jobs)
			for job in model_jobs_copy:
				_, _, status = get_job_info(job['job_id'])

				

				if status == 'FAILED' or status.startswith('CANCELLED'):
					# Assume job was unsuccessful
					model_jobs.remove(job)
					shutil.rmtree(os.path.join(pretrain_dir, job['model_hash']), ignore_errors=True)
					shutil.rmtree(os.path.join(finetune_dir, job['model_hash']), ignore_errors=True)

			print(f'{pu.bcolors.WARNING}Removed failed jobs{pu.bcolors.ENDC}')
			wait_for_jobs(model_jobs)
	else:
		os.makedirs(f'./job_scripts/{args.task}', exist_ok=True)

	trained_hashes, pipeline_hashes = [], []
	for job in model_jobs:
		_, _, status = get_job_info(job['job_id'])
		if status == 'COMPLETED':
			trained_hashes.append(job['model_hash'])
		else:
			pipeline_hashes.append(job['model_hash'])

	pretrained_hashes = os.listdir(pretrain_dir)
	if os.path.exists(model_jobs_file):
		pretrained_hashes = []
		for model_hash in os.listdir(pretrain_dir):
			if model_hash in trained_hashes:
				pretrained_hashes.append(model_hash)

	# Finetune pretrained models
	for model_hash in pretrained_hashes:
		if os.path.exists(finetune_dir) and model_hash in os.listdir(finetune_dir):
			continue

		model, model_idx = graphLib.get_graph(model_hash=model_hash)
		
		job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
			epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, dataset_file=args.dataset_file,
			autotune=args.autotune, autotune_trials=autotune_trials, cluster=args.cluster, id=args.id)
		assert pretrain is False
		
		model_jobs.append({'model_hash': model.hash, 
			'job_id': job_id, 
			'train_type': 'P+F' if pretrain else 'F'})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Save model jobs
	json.dump(model_jobs, open(model_jobs_file, 'w+'))

	# Pretrain and then finetune randomly sampled models if total finetuned models are less than num_init
	# TODO: Add skopt.sampler.Sobol points instead
	while len(pretrained_hashes) < args.num_init:
		sample_idx = random.randint(0, len(graphLib))
		model_hash = graphLib.library[sample_idx].hash

		if model_hash not in pretrained_hashes:
			pretrained_hashes.append(model_hash)

			model, model_idx = graphLib.get_graph(model_hash=model_hash)

			job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
				epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, dataset_file=args.dataset_file,
				autotune=args.autotune, autotune_trials=autotune_trials, cluster=args.cluster, id=args.id)
			assert pretrain is True
			
			model_jobs.append({'model_hash': model.hash, 
				'job_id': job_id, 
				'train_type': 'P+F' if pretrain else 'F'})

	# Wait for jobs to complete
	wait_for_jobs(model_jobs)

	# Save model jobs
	json.dump(model_jobs, open(model_jobs_file, 'w+'))

	# Update dataset with newly trained models
	old_best_performance = update_dataset(graphLib, args.task, finetune_dir, new_dataset_file)

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

	# Train BOSHNAS model based on trained models
	y = y/max_loss
	train_error = surrogate_model.train(X, y)

	same_performance = 0
	method = 'optimization'

	while same_performance < ACCURACY_PATIENCE + ALEATORIC_QUERIES:
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

		if method == 'optimization':
			print(f'{pu.bcolors.OKBLUE}Running optimization step{pu.bcolors.ENDC}')
			
			# Use aleatoric loss close to convergence to optimize training recipe
			if same_performance < ACCURACY_PATIENCE:
				# Architecture not converged yet. Use only epistemic uncertainty
				use_al = False
			else:
				# Use aleatoric uncertainty to optimize training recipe
				use_al = True
				autotune_trials = 2 * args.autotune_trials
				print(f'{pu.bcolors.OKBLUE}Aleatoric uncertainty being used{pu.bcolors.ENDC}')

			# Get next queries
			query_indices = surrogate_model.get_queries(x=X_ds, k=K, explore_type='ucb', use_al=use_al)

			# Run queries
			for i in set(query_indices):
				model = graphLib.library[i]

				if not use_al and model.hash in trained_hashes + pipeline_hashes:
					# If aleatoric uncertainty is not considered, only consider models that are not 
					# already trained or in the pipeline
					continue

				chosen_neighbor_hash = get_neighbor_hash(model, graphLib, trained_hashes)

				if chosen_neighbor_hash:
					# Finetune model with the chosen neighbor
					job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
						epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=chosen_neighbor_hash, 
						dataset_file=args.dataset_file, autotune=args.autotune, autotune_trials=autotune_trials,
						cluster=args.cluster, id=args.id)
					assert pretrain is False
				else:
					# If no neighbor was found with high overlap, proceed to next query model
					continue

				new_queries += 1
				
				model_jobs.append({'model_hash': model.hash, 
					'job_id': job_id, 
					'train_type': 'P+F' if pretrain else 'F'})
			
			if new_queries == 0:
				# If no queries were found where direct finetuning could be performed, pretrain model
				# with best acquisition function value
				query_embeddings = [X_ds[idx, :] for idx in set(query_indices)]
				candidate_predictions = surrogate_model.predict(query_embeddings)

				best_prediction_indices = [query_indices[idx] for idx in np.argsort(acq([pred[0] for pred in candidate_predictions],
											[pred[1][0] + pred[1][1] for pred in candidate_predictions],
											explore_type='ucb'))]
				
				if len(best_prediction_indices) > 1:
					for best_prediction_index in best_prediction_indices:
						model = graphLib.library[best_prediction_index]
						if use_al or model.hash not in trained_hashes + pipeline_hashes:
							# If model already trained, take the next best model
							break
				else:
					model = graphLib.library[best_prediction_indices[0]]
					for neighbor_hash in model.neighbors:
						neighbor_model, _ = graphLib.get_graph(model_hash=neighbor_hash)
						if use_al or neighbor_model.hash not in trained_hashes + pipeline_hashes:
							# If model already trained, take the nearest untrained neighbor
							break
					model = neighbor_model

				# Pretrain model
				job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
					epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, dataset_file=args.dataset_file,
					autotune=args.autotune, autotune_trials=autotune_trials, cluster=args.cluster, id=args.id)

				new_queries += 1
				
				model_jobs.append({'model_hash': model.hash, 
					'job_id': job_id, 
					'train_type': 'P+F' if pretrain else 'F'})

		elif method == 'unc_sampling':
			print(f'{pu.bcolors.OKBLUE}Running uncertainty sampling{pu.bcolors.ENDC}')

			candidate_predictions = surrogate_model.predict(X_ds)

			# Get model index with highest epistemic uncertainty
			unc_prediction_idx = np.argmax([pred[1][0] for pred in candidate_predictions])

			# Sanity check: model with highest epistemic uncertainty should not be trained
			assert graphLib.library[unc_prediction_idx].hash not in trained_hashes, \
				'model with highest uncertainty was found trained'

			if graphLib.library[unc_prediction_idx].hash in pipeline_hashes:
				print(f'{pu.bcolors.OKBLUE}Highest uncertainty model already in pipeline{pu.bcolors.ENDC}')
			else:
				model = graphLib.library[unc_prediction_idx]

				# Pretrain sampled architecture
				job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
						epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, dataset_file=args.dataset_file,
						autotune=args.autotune, autotune_trials=autotune_trials, cluster=args.cluster, id=args.id)
				assert pretrain is True

				new_queries += 1
				
				model_jobs.append({'model_hash': model.hash, 
					'job_id': job_id, 
					'train_type': 'P+F' if pretrain else 'F'})

		else:
			print(f'{pu.bcolors.OKBLUE}Running diversity sampling{pu.bcolors.ENDC}')

			# Get randomly sampled model idx
			# TODO: Add skopt.sampler.Sobol points instead
			div_prediction_idx = random.randint(0, len(graphLib))

			while graphLib.library[div_prediction_idx].hash in trained_hashes + pipeline_hashes \
				or (HOMOGENEOUS_ONLY and not is_homogenous(graphLib.library[div_prediction_idx])):
				div_prediction_idx = random.randint(0, len(graphLib))

			model = graphLib.library[div_prediction_idx]

			# Pretrain sampled architecture
			job_id, pretrain = worker(model_dict=model.model_dict, model_hash=model.hash, task=args.task, 
					epochs=args.epochs, models_dir=args.models_dir, chosen_neighbor_hash=None, dataset_file=args.dataset_file,
					autotune=args.autotune, autotune_trials=autotune_trials, cluster=args.cluster, id=args.id)
			# assert pretrain is True

			new_queries += 1
			
			model_jobs.append({'model_hash': model.hash, 
				'job_id': job_id, 
				'train_type': 'P+F' if pretrain else 'F'})

		# Wait for jobs to complete
		wait_for_jobs(model_jobs)

		# Save model jobs
		json.dump(model_jobs, open(model_jobs_file, 'w+'))

		# Get current tabular dataset
		X, y = convert_to_tabular(graphLib)
		y = y/max_loss

		# Train BOSHNAS model based on new trained queries
		train_error = surrogate_model.train(X, y)

		# Update dataset with newly trained models
		best_performance = update_dataset(graphLib, args.task, finetune_dir, new_dataset_file)

		# Update same_performance to check convergence
		if best_performance == old_best_performance and method == 'optimization':
			same_performance += 1
		else:
			same_performance = 0

		old_best_performance = best_performance

	# Wait for jobs to complete
	wait_for_jobs(model_jobs, running_limit=0, patience=0)

	# Save model jobs
	json.dump(model_jobs, open(model_jobs_file, 'w+'))

	# Update dataset with newly trained models
	best_performance = update_dataset(graphLib, args.task, finetune_dir, new_dataset_file)

	print(f'{pu.bcolors.OKGREEN}Convergence criterion met!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	main()

