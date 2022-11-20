import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import torch

from transformers import BertConfig,  BertModel, BertTokenizer

from transformers.models.flexibert.modeling_flexibert import FlexiBERTConfig, FlexiBERTModel

from library import Graph, GraphLib
from utils import print_util as pu


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

graphLib = GraphLib.load_from_dataset('../dataset/dataset_test.json')


# Bert-Mini

bert_mini = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')


model_dict_bert_mini = {'l': 4, 'o': ['sa']*4, 'h': [256]*4, 'n': [4]*4, 'f': [[1024]]*4, 'p': ['sdp']*4}

config = FlexiBERTConfig()

config.from_model_dict(model_dict_bert_mini)

model_modular = FlexiBERTModel(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_mini.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_mini_graph = graphLib.get_graph(model_dict=model_dict_bert_mini)[0]

if bert_mini_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-Mini found in dataset!{pu.bcolors.ENDC}')
	print(bert_mini_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_mini_graph.hash)+'/')

tokenizer.save_pretrained("../models/pretrained/"+str(bert_mini_graph.hash)+'/')

print("BERT-Mini heterogenized")


# Bert-Tiny

bert_tiny = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

model_dict_bert_tiny= { 'l': 2, 'o': ['sa']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['sdp']*2}

config = FlexiBERTConfig()

config.from_model_dict(model_dict_bert_tiny)

model_modular = FlexiBERTModel(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_tiny.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_tiny_graph = graphLib.get_graph(model_dict=model_dict_bert_tiny)[0]

if bert_tiny_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-Tiny found in dataset!{pu.bcolors.ENDC}')
	print(bert_tiny_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_tiny_graph.hash)+'/')

tokenizer.save_pretrained("../models/pretrained/"+str(bert_tiny_graph.hash)+'/')

print("BERT-Tiny heterogenized")


# Bert-L2-H256-A4

bert_2_256 = BertModel.from_pretrained('google/bert_uncased_L-2_H-256_A-4')

model_dict_bert_2_256= { 'l': 2, 'o': ['sa']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['sdp']*2}

config = FlexiBERTConfig()

config.from_model_dict(model_dict_bert_2_256)

model_modular = FlexiBERT(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_2_256.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_2_256_graph = graphLib.get_graph(model_dict=model_dict_bert_2_256)[0]

if bert_2_256_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-2-256 found in dataset!{pu.bcolors.ENDC}')
	print(bert_2_256_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_2_256_graph.hash)+'/')


tokenizer.save_pretrained("../models/pretrained/"+str(bert_2_256_graph.hash)+'/')

print("Bert L-2/H-256/A-4 heterogenized")


# Bert-L4-H128-A2

bert_4_128 = BertModel.from_pretrained('google/bert_uncased_L-4_H-128_A-2')

model_dict_bert_4_128= { 'l': 4, 'o': ['sa']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['sdp']*4}

config = FlexiBERTConfig()

config.from_model_dict(model_dict_bert_4_128)

model_modular = FlexiBERTModel(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_4_128.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_4_128_graph = graphLib.get_graph(model_dict=model_dict_bert_4_128)[0]

if bert_4_128_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-4-128 found in dataset!{pu.bcolors.ENDC}')
	print(bert_4_128_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_4_128_graph.hash)+'/')

tokenizer.save_pretrained("../models/pretrained/"+str(bert_4_128_graph.hash)+'/')

print("Bert L-4/H-128/A-2 heterogenized")

