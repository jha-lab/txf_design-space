# Run a Gaussian-process based active-learning framework to build
# a surrogate model for all the transformer architectures in the
# design space.

# Author : Shikhar Tuli

import os
import sys
sys.path.append('../../transformers/src/')
sys.path.append('../../embeddings/')
sys.path.append('../')

import logging
# logging.disable(logging.INFO)

import argparse
from multiprocessing import Process, Manager, Pool, Queue
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import numpy as np
import pickle
import time
import random
import pdb

import torch
from transformers import  BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular

from finetune_flexibert import finetune
import shlex

from library import GraphLib, Graph
from utils import print_util as pu


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
GLUE_TASKS_DATASET = ['CoLA', 'MNLI-mm', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']

CONF_INTERVAL = 0.001 # Corresponds to 0.1% accuracy for 95% confidence interval
OVERLAP_THRESHOLD = 0.9 # Corresponds to the minimum overlap for model to be considered

DEBUG = False


def worker(worker_id: int, 
    shared_accuracies: list, 
    model_idx: int, 
    chosen_neighbor: str,
    model_dict, 
    model_hash: str, 
    task: str, 
    models_dir: str):
    """Worker to fine-tune the given model
    
    Args:
        worker_id (int): wroker index in the node, should be from 0 to n_jobs
        shared_accuracies (list): shared accuracies for all workers
        model_idx (int): index for the model in shared_accuracies
        from_neighbor (bool): True if model was loaded from a fine-tuned neighbor
        model_hash (str): hash of the given model
        task (str): name of the GLUE task for fine-tuning the model on; should
            be in GLUE_TASKS
        models_dir (str): path to "models" directory containing "pretrained" sub-directory
    """
    # Forcing to train on single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    model_config = BertConfig()
    model_config.from_model_dict(model_dict)
    chosen_neighbor_model = BertModelModular.from_pretrained(
        f'{models_dir}{task}/{chosen_neighbor}/')
    current_model = BertModelModular(model_config)
    current_model.load_model_from_source(chosen_neighbor_model)
    current_model.save_pretrained(
        f'{models_dir}{task}/{model_hash}/')

    model_name_or_path = f'{models_dir}{task}/{model_hash}/' 

    # Initialize training arguments for fine-tuning
    training_args = f'--model_name_or_path {model_name_or_path} \
        --task_name {task} \
        --do_train \
        --do_eval \
        --save_total_limit 2 \
        --max_seq_length 128 \
        --per_device_train_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --overwrite_output_dir \
        --output_dir {models_dir}{task}/{model_hash}/'

    training_args = shlex.split(training_args)

    if not DEBUG:
        # Fine-tune current model
        metrics = finetune(training_args)

        # Add accuracy to graph library
        shared_accuracies[model_idx] = metrics['eval_accuracy'] 
    else:
        # Random time for fine-tuning
        time.sleep(random.randint(50, 90))

        # Add random accuracy to graph library
        shared_accuracies[model_idx] = random.uniform(0.6, 0.95)
     
def main():   
    task = 'sst2'

    dataset_file = '../../dataset/dataset_small.json'

    # New dataset file
    new_dataset_file = dataset_file.split('.json')[0] + f'_{task}.json'

    # Load new GraphLib object from updated dataset
    graphLib = GraphLib.load_from_dataset(new_dataset_file)

    manager = Manager()
    shared_accuracies = manager.list([None for _ in range(len(graphLib))])

    # Re-load shared accuracies
    for i in range(len(graphLib)):
        shared_accuracies[i] = graphLib.library[i].accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(task)]]


    model_idx = 178
    chosen_neighbor = 'c8dba6c37045e01844bfa62c8570ec6d'
    worker_id_free = 0
    models_dir = '../../models/'


    # model_config = BertConfig()
    # model_config.from_model_dict(graphLib.library[model_idx].model_dict)
    # chosen_neighbor_model = BertModelModular.from_pretrained(
    #     f'{models_dir}{task}/{chosen_neighbor}/')
    # current_model = BertModelModular(model_config)
    # current_model.load_model_from_source(chosen_neighbor_model)
    # current_model.save_pretrained(
    #     f'{models_dir}{task}/{graphLib.library[model_idx].hash}/')

    # worker(worker_id_free, 
    #     shared_accuracies, 
    #     model_idx, 
    #     True, 
    #     '939c7b996fb1cd0d9918a309caed3405', # graphLib.library[model_idx].hash,
    #     args.task, 
    #     args.models_dir)

    proc = Process(target=worker, args=(worker_id_free, 
                                shared_accuracies,
                                model_idx,
                                'c8dba6c37045e01844bfa62c8570ec6d',
                                graphLib.library[model_idx].model_dict,
                                '939c7b996fb1cd0d9918a309caed3405', # graphLib.library[model_idx].hash,
                                task,
                                models_dir))
    proc.daemon = True
    proc.start()
    proc.join()

if __name__ == '__main__':
    main()