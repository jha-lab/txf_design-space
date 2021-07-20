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


def returntrainingargs(models_dir,task,model_hash):

	model_name_or_path = f'{models_dir}pretrained/{model_hash}'

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

    return training_args

def main():

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
    parser.add_argument('--model_dict',
    	metavar='',
    	type=dict,
    	help='model dict of the pretrained model'
    	)



    args = parser.parse_args()
   	graphLib = GraphLib.load_from_dataset(args.dataset_file)
   	graph = graphLib.get_graph(model_dict=args.model_dict)[0]
   	
   	glue_scores = {}

   	for task in GLUE_TASKS:

   		training_args = returntrainingargs(args.models_dir, task, graph.hash)
   		metrics = finetune(training_args)

   		if task == 'cola':

   			glue_scores[task] = metrics['eval_matthews_correlation']

   		elif task == 'stsb':

   			glue_scores[task+'_spearman'] = metrics['eval_spearmanr']
   			glue_scores[task+'_pearson'] = metrics['eval_pearson']

   		elif task == 'mrpc' or task=='qqp':

   			glue_scores[task+'_accuracy'] = metrics['eval_accuracy']
   			glue_scores[task+'_f1'] = metrics['eval_f1']

   		elif task in ["sst2", "mnli",  "qnli", "rte", "wnli"]:

   			glue_scores[task] = metrics['eval_accuracy']

   	score = 0
   	total = 0
   	
   	for key, value in glue_scores:

   		score+=value
   		total +=1
   		print(key,':',value)

   	print('SCORE:', score*1.0/total)

   	output_dir = f"{models_dir}glue_scores/{graph.hash}.json"

   	with open(output_dir, 'w') as fp:
    	son.dump(glue_scores, fp)














