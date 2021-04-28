# Run a Gaussian-process based active-learning framework to build
# a surrogate model for all the transformer architectures in the
# design space.

# Author : Shikhar Tuli

import os
import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import logging
logging.disable(logging.INFO)

import argparse
from multiprocessing import Process, Manager
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import numpy as np
import pickle
import time
import random

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

DEBUG = True


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

    # Initialize training arguments for fine-tuning
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
        help='number of parallel jobs (GPU cores); not used right now',
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
    
    if not DEBUG:
        # Instantiate processes list
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
    new_dataset_file = args.dataset_file.split('.json')[0] + f'_{args.task}.json'
    graphLib = GraphLib.load_from_dataset(new_dataset_file)

    # Re-load shared accuracies
    for i in range(len(graphLib)):
        shared_accuracies[i] = graphLib.library[i].accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(args.task)]]

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

    if not DEBUG:
        # Save the fitted model
        if not os.path.exists('../dataset/surrogate_models/'):
        	os.mkdir('../dataset/surrogate_models/')
        with open(args.surrogate_model_file, 'wb') as surrogate_model_file:
            pickle.dump(surrogate_model, surrogate_model_file)

        print(f'{pu.bcolors.OKGREEN}Surrogate model saved to:{pu.bcolors.ENDC} {args.surrogate_model_file}')

    # Pseudo-code for active-learning
    # 1. Take the argmax (or argsort) of standard deviation from the GP model (will have to 
    # 	use np.meshgrid) and then find the nearest 8-dimensional embedding of the transformer. 
    # 2. Find the overlap of this model with the nearest 10 neighbors. If overlap is less 
    # 	than 0.90, take next most uncertain model (using argsort of standard deviation).
    # 3. Repeat above two steps asynchronously as-soon-as one model has been trained in the pool
    # 	of four workers in the cluster node.
    # 4. Convergence condition: Once 1.96 * max(standard deviation) < 0.001 (i.e. 95% confidence 
    # 	interval is less than 0.1% accuracy).
    # # TODO: (extras) hyper-parameter tuning for every new model using ray-tune; reinforcement 
    # 	learning for faster convergence; use better modeling technique than GP
    
    # Get all input points for the design space
    X_ds = np.zeros((len(graphLib), embedding_size))
    for i in range(len(graphLib)):
        X_ds[i, :] = graphLib.library[i].embedding

    # Get all predicted accuracies with standard deviation for all points in the design space
    y_ds, std_ds = surrogate_model.predict(X_ds, return_std=True)

    # Create a dictionary of workers, every worker points to a tuple of model
    # index and process pointer
    jobs = {0: (None, None), 1: (None, None), 2: (None, None), 3: (None, None)} 

    while 1.96 * np.amax(std_ds) > CONF_INTERVAL:
        # Wait till a worker is free
        worker_id_free = None
        while worker_id_free is None:
            for worker_id in jobs:
                if jobs[worker_id][1] is None or not jobs[worker_id][1].is_alive():
                    # if jobs[worker_id][1] is not None: jobs[worker_id][1].join()
                    worker_id_free = worker_id
                    jobs[worker_id] = (None, None)
            time.sleep(1)

        # Update the number of models trained
        print(f'{pu.bcolors.OKGREEN}Number of models trained in the design space:{pu.bcolors.ENDC} ' \
            + f'{sum(acc is not None for acc in shared_accuracies)}')
        print()

        # Update graphLib object with new accuraies
        for i in range(len(shared_accuracies)):
            graphLib.library[i].accuracy[GLUE_TASKS_DATASET[GLUE_TASKS.index(args.task)]] = shared_accuracies[i]

        # Update surrogate model
        trained_ids = [idx for idx, acc in enumerate(shared_accuracies) if acc is not None]
        surrogate_model.fit(X_ds[trained_ids, :], np.array(shared_accuracies)[trained_ids])

        # Update uncertainties (standard deviation) in the model
        y_ds, std_ds = surrogate_model.predict(X_ds, return_std=True)

        # Print current max of standard deviation
        print(f'{pu.bcolors.OKGREEN}Current max of standard deviation:{pu.bcolors.ENDC} {np.amax(std_ds): 0.4f}')
        print()

        # Get model indices for determining next queried architecture
        model_ids = np.argsort(std_ds)[::-1]

        # Get next model_idx to be queried for training
        model_idx = -1
        for i in range(len(graphLib)):
            model_idx = model_ids[i]

            print(f'Checking model index: {model_idx}')

            # Check if this model is already in training
            if model_idx in [worker[0] for worker_id, worker in jobs.items()]:
                continue

            # Check overlap with neighors
            #overlap, trained = [], []

            is_test = False
            max_overlap = 0

            for neighbor in graphLib.library[model_idx].neighbors:
                neighbor_graph, neighbor_idx = graphLib.get_graph(model_hash=neighbor)

                config = BertConfig()
                config.from_model_dict(graphLib.library[model_idx].model_dict)
                current_model = BertModelModular(config)

                config = BertConfig()
                config.from_model_dict(neighbor_graph.model_dict)
                neighbor_model = BertModelModular(config)

                overlap = current_model.load_model_from_source(neighbor_model)

                if overlap > OVERLAP_THRESHOLD and (neighbor_idx in trained_ids):

                    is_test = True

                    if overlap > max_overlap:
                        

                        max_overlap = overlap
                        #Save pretrained model of the current model after loading weights from finetuned model of the neighbor
                        nbd_model = BertModelModular.from_pretrained(f'{args.models_dir}{args.task}/{neighbor}/')
                        current_model.load_model_from_source(nbd_model)
                        current_model.save_pretrained(f'{args.models_dir}pretrained/{graphLib.library[model_idx].hash}/')


            # Effective overlap is the overlap between the current model and its neighbor if
            # the neighbor has been trained
            #effective_overlap = [overlap[i] if trained[i] == True else 0 for i in range(len(overlap))]

            # Test current model only if effective overlap with any one of the neighbors is greater
            # than the overlap threshold
            #test_model_idx = [True if ef_ov > OVERLAP_THRESHOLD else False for ef_ov in effective_overlap]



            if not DEBUG:
                if is_test:
                    # Choose current model index
                    break
            else:
                # Choose this model index as the next query regardless of overlap
                break

        if model_idx == -1:
            if 1.96 * np.amax(std_ds) > CONF_INTERVAL:
                raise ValueError('No model found for next iteration even when convergence has not reached!')
            else:
                print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')
                break

        print(f'{pu.bcolors.OKBLUE}Training model:{pu.bcolors.ENDC}\n{graphLib.library[model_idx]}')
        print()

        proc = Process(target=worker, args=(worker_id_free, 
                                        shared_accuracies,
                                        model_idx,
                                        graphLib.library[model_idx].hash,
                                        args.task,
                                        args.models_dir))
        jobs[worker_id_free] = (model_idx, proc)
        proc.start()

        print(f'{pu.bcolors.WARNING}Jobs dictionary:{pu.bcolors.ENDC} {jobs}')
        print()

    print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')
    print()

    if not DEBUG:
        # Save the fitted model
        if not os.path.exists('../dataset/surrogate_models/'):
            os.mkdir('../dataset/surrogate_models/')
        with open(args.surrogate_model_file, 'wb') as surrogate_model_file:
            pickle.dump(surrogate_model, surrogate_model_file)

        print(f'{pu.bcolors.OKGREEN}Surrogate model saved to:{pu.bcolors.ENDC} {args.surrogate_model_file}')


if __name__ == '__main__':
    main()



