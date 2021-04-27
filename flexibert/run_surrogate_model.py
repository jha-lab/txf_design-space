# Run a Gaussian-process based active-learning framework to build
# a surrogate model for all the transformer architectures in the
# design space.

# Author : Shikhar Tuli

import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import logging
logging.basicConfig(level=logging.ERROR)

import argparse
from multiprocessing import Process, Manager
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import numpy as np
import pickle

import torch
from transformers import  BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular

from finetune_flexibert import finetune
import shlex

from library import GraphLib, Graph
from utils import print_util as pu


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
GLUE_TASKS_DATASET = ['CoLA', 'MNLI-mm', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']


def worker(worker_id: int, shared_accuracies: list, model_idx: int, model_hash: str, task: str, models_dir: str):
    """Worker to fine-tune the given model
    
    Args:
        worker_id (int): wroker index in the node, should be from 0 to n_jobs
        shared_accuracies (list): shared accuracies for all workers
        model_idx (int): index for the model in shared_accuracies
        model_hash (str): hash of the given model
        task (str): name of the GLUE task for fine-tuning the model on; should
            be in GLUE_TASKS
        models_dir (str): path to "models" directory containing "pretrained" sub-directory
    """
    # Forcing to train on single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

    Initialize training arguments for fine-tuning
    training_args = f'--model_name_or_path {models_dir}pretrained/{model_hash}/ \
        --task_name {task} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --overwrite_output_dir \
        --output_dir {models_dir}{task}/{model_hash}/'

    training_args = shlex.split(training_args)

    # Fine-tune current model
    metrics = finetune(training_args)

    # Add accuracy to graph library
    shared_accuracies[model_idx] = metrics['eval_accuracy'] 


def main():
    """Run active-learning framework for training models in the design space
    """
    parser = argparse.ArgumentParser(
        description='Input parameters for generation of dataset library',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_file',
        metavar='',
        type=str,
        help='path to load the dataset',
        default='../dataset/dataset_small.json')
    parser.add_argument('--surrogate_model_file',
        metavar='',
        type=str,
        help='path to save the surrogate model parameters',
        default='../dataset/surrogate_models/gp_sst2.pkl')
    parser.add_argument('--models_dir',
        metavar='',
        type=str,
        help='path to "models" directory containing "pretrained" sub-directory',
        default='../models/')
    parser.add_argument('--task',
        metavar='',
        type=str,
        help=f'name of GLUE tasks to train surrogate model for',
        default='sst2')
    parser.add_argument('--n_jobs',
        metavar='',
        type=int,
        help='number of parallel jobs (GPU cores); not usable right now',
        default=4)

    args = parser.parse_args()

    random_seed = 1

    assert args.task in GLUE_TASKS, f'GLUE task should be in: {GLUE_TASKS}'

    # Instantiate GraphLib object
    graphLib = GraphLib.load_from_dataset(args.dataset_file)

    # First fine-tune all pre-trained models and initialize surrogate model
    pretrained_model_hashes = os.listdir(args.models_dir + 'pretrained/')

    # Maintain a list of shared accuracies (among all workers) for the given task
    manager = Manager()
    shared_accuracies = manager.list([None for _ in range(len(graphLib))])

    #Instantiate processes list
    procs = []

    # Fine-tune four pretrained models in the design space
    print(f'{pu.bcolors.OKBLUE}Fine-tuning four pretrained models in the design space{pu.bcolors.ENDC}')
    for i in range(len(pretrained_model_hashes)):
    	_, model_idx = graphLib.get_graph(model_hash=pretrained_model_hashes[i])
    	proc = Process(target=worker, args=(i, 
                                        shared_accuracies,
                                        model_idx,
                                        pretrained_model_hashes[i],
                                        args.task,
                                        args.models_dir))
    	procs.append(proc)
    	proc.start()

    # Join finished processes
    for proc in procs:
        proc.join()

    # Save accuracies to GraphLib object
    for i in range(len(shared_accuracies)):
    	graphLib.library[i].accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(args.task)]] = shared_accuracies[i]

    # Save new GraphLib object into dataset file
    new_dataset_file = args.dataset_file.split('.json')[0] + f'_{args.task}.json'
    graphLib.save_dataset(new_dataset_file)
    print()
    
    # Load new GraphLib object from updated dataset
    graphLib = GraphLib.load_from_dataset(new_dataset_file)

    # Initialize the surrogate model with fine-tuned models
    surrogate_model = GP(n_restarts_optimizer=10, random_state=random_seed)

    # Get input and output for training surrogate model
    embedding_size = graphLib.library[0].embedding.shape[0]
    X = np.zeros((len(pretrained_model_hashes), embedding_size))
    y = np.zeros(len(pretrained_model_hashes))
    for i in range(len(pretrained_model_hashes)):
        graph, _ = graphLib.get_graph(model_hash=pretrained_model_hashes[i])
        X[i, :] = graph.embedding
        y[i] = graph.accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(args.task)]]

    # Fit surrogate model
    print(f'{pu.bcolors.OKBLUE}Fitting surrogate model{pu.bcolors.ENDC}')
    surrogate_model.fit(X, y)

    # Print coefficent of determination
    print(f'{pu.bcolors.OKGREEN}Coefficent of determination after ' \
    	+ f'fitting surrogate model:{pu.bcolors.ENDC} {surrogate_model.score(X, y): 0.4f}')

    # Save the fitted model
    if not os.path.exists('../dataset/surrogate_models/'):
    	os.mkdir('../dataset/surrogate_models/')
    with open(args.surrogate_model_file, 'wb') as surrogate_model_file:
        pickle.dump(surrogate_model, surrogate_model_file)

    print(f'{pu.bcolors.OKGREEN}Surrogate model saved to:{pu.bcolors.ENDC} {args.surrogate_model_file}')


if __name__ == '__main__':
    main()




