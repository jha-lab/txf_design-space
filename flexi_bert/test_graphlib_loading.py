# Tests GraphLib class with neighborhood feature and weight
# transfer into the queried graph

# Author : Shikhar Tuli

import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import torch
from transformers import  BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular
from library import Graph, GraphLib
from utils import print_util as pu


# Testing if bert-mini is in design_space_small
model_dict_bert_mini = {'l':4, 'a':[4]*4, 'f':[4*256]*4, 'h':[256]*4, 's':['sdp']*4}

graphLib = GraphLib.load_from_dataset('../dataset/dataset_small.json')
bert_mini_graph = graphLib.get_graph(model_dict=model_dict_bert_mini)

if bert_mini_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-Mini found in dataset!{pu.bcolors.ENDC}')
	print(bert_mini_graph, '\n')

# Creating bert-mini model
config = BertConfig()
config.from_model_dict(model_dict_bert_mini)
bert_mini = BertModelModular(config)
bert_mini.load_state_dict(torch.load('../main_models/bert_mini.pth'))

print(f'{pu.bcolors.OKGREEN}BERT-Mini model loaded!{pu.bcolors.ENDC}\n')

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

neighbor_idx_max_overlap = 0
max_overlap = 0

for n in range(graphLib.num_neighbors):
	# Get neighbor of bert-mini in design space
	bert_mini_neighbor_graph = graphLib.get_graph(neighbor_hash=bert_mini_graph.neighbors[n])

	if bert_mini_neighbor_graph is not None:
		print(f'{pu.bcolors.OKGREEN}BERT-Mini\'s {ordinal(n+1)} neighbor found in dataset!{pu.bcolors.ENDC}')
		print(bert_mini_neighbor_graph, '\n') # prints representation of the neighbor graph

	# Test loading of weights for bert-mini's neighbor
	config = BertConfig()
	config.from_model_dict(bert_mini_neighbor_graph.model_dict)
	bert_mini_neighbor = BertModelModular(config)
	overlap = bert_mini_neighbor.load_model_from_source(bert_mini)

	if overlap > max_overlap:
		neighbor_idx_max_overlap = n
		max_overlap = overlap

	print(f'{pu.bcolors.OKBLUE}Overlap with BERT-Mini: {overlap: 0.4f}{pu.bcolors.ENDC}\n')

print(f'{pu.bcolors.OKGREEN}Nearest neighbor with max overlap of {max_overlap}:{pu.bcolors.ENDC}')
print(graphLib.get_graph(neighbor_hash=bert_mini_graph.neighbors[neighbor_idx_max_overlap]))

