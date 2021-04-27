# Run a Gaussian-process based active-learning framework to build
# a surrogate model for all the transformer architectures in the
# design space.

# Author : Shikhar Tuli

import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import argparse
from multiprocessing import Process
from sklearn.GaussianProcess import GaussianProcess
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


def worker(worker_id: int, graphLib: 'GraphLib', model_hash: str, task: str, models_dir: str):
    """Worker to fine-tune the given model
    
    Args:
        worker_id (int): wroker index in the node, should be from 0 to n_jobs
        graphLib (GraphLib): graph library object to be used for modeling
        model_hash (str): hash of the given model
        task (str): name of the GLUE task for fine-tuning the model on; should
            be in GLUE_TASKS
        models_dir (str): path to "models" directory containing "pretrained" sub-directory
    """
    # Forcing to train on single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = worker_id

    # Initialize training arguments for fine-tuning
    training_args = f'--model_name_or_path {models_dir + 'pretrained/' + model_hash + '/'} \
        --task_name {task} \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --overwrite_output_dir \
        --output_dir {models_dir + task + '/' + model_hash + '/'}'

    training_args = shlex.split(training_args)

    # Fine-tune current model
    metrics = finetune(training_args)

    # Add accuracy to graph library
    model_graph = graphLib.get_graph(model_hash=model_hash)
    model_graph.accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(task)]] = metrics['eval_accuracy']


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
        default='../dataset/surrogate_models/gp_sst2')
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
        help='number of parallel jobs (GPU cores)',
        default=4)

    args = parser.parse_args()

    assert args.task in GLUE_TASKS, f'GLUE task should be in: {GLUE_TASKS}'

    # Instantiate GraphLib object
    graphLib = GraphLib.load_from_dataset(args.dataset_file)

    # First fine-tune all pre-trained models and initialize surrogate model
    pretrained_model_hashes = os.listdir(args.models_dir + 'pretrained/')

    #Instantiate processes list
    procs = []

    # Fine-tune four pretrained models in the desing space
    for i in range(len(pretrained_model_hashes)):
        proc = Process(target=worker, args=(i, 
                                        graphLib,
                                        pretrained_model_hashes[i],
                                        args.task,
                                        args.models_dir))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    # Save new GraphLib object into dataset file
    graphLib.save_dataset(args.dataset_file.split('.json')[0] + f'_{args.task}.json')

    # Initialize the surrogate model with fine-tuned models
    surrogate_model = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4,
                                thetaU=1e-1, random_start=100, normalize=False)

    # Get input and output for training surrogate model
    embedding_size = graphLib.library[0].embedding.shape[0]
    X = np.zeros((4, embedding_size))
    y = np.zeros(4)
    for i in range(len(pretrained_model_hashes)):
        graph = graphLib.get_graph(model_hash=pretrained_model_hashes[i])
        X[i, :] = graph.embedding
        y[i] = graph.accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(args.task)]]

    # Fit surrogate model
    surrogate_model.fit(X, y)

    # Save the fitted model
    with open(args.surrogate_model_file, 'wb') as surrogate_model_file:
        pickle.dump(surrogate_model, surrogate_model_file)



if __name__ == '__main__':
    main()




