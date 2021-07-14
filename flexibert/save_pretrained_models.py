import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import torch

from transformers import  BertConfig,  BertModel, BertTokenizer

from transformers.models.bert.modeling_modular_bert import BertModelModular

from library import Graph, GraphLib
from utils import print_util as pu


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

graphLib = GraphLib.load_from_dataset('../dataset/dataset_test.json')

#BertBase 
'''
bert_base = BertModel.from_pretrained('bert-base-uncased')

model_dict_bert_base = {'l':12,'a':[8]*12,'f':[4*768]*12,'h':[768]*12,'s':['sdp']*12}

config = BertConfig()

config.from_model_dict(model_dict_bert_base)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_base.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_base_graph = graphLib.get_graph(model_dict=model_dict_bert_base)

if bert_base_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-Base found in dataset!{pu.bcolors.ENDC}')
	print(bert_base_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_base_graph.hash)+'/')

#torch.save(modular_state_dict, '../main_models/bert_base.pth')

print("Bert Base Modularized")

#BertSmall


bert_small = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8')

model_dict_bert_small = {'l':4,'a':[8]*4,'f':[4*512]*4,'h':[512]*4,'s':['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_small)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_small.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

#torch.save(modular_state_dict, '../main_models/bert_small.pth')

bert_small_graph = graphLib.get_graph(model_dict=model_dict_bert_small)

if bert_small_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-Small found in dataset!{pu.bcolors.ENDC}')
	print(bert_small_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_small_graph.hash)+'/')

#torch.save(modular_state_dict, '../main_models/bert_base.pth')

print("Bert Small Modularized")

#BertMedium


bert_medium = BertModel.from_pretrained('google/bert_uncased_L-8_H-512_A-8')

model_dict_bert_medium = {'l':8,'a':[8]*8,'f':[4*512]*8,'h':[512]*8,'s':['sdp']*8}

config = BertConfig()

config.from_model_dict(model_dict_bert_medium)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_medium.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_medium_graph = graphLib.get_graph(model_dict=model_dict_bert_medium)

if bert_medium_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-medium found in dataset!{pu.bcolors.ENDC}')
	print(bert_medium_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_medium_graph.hash)+'/')


#torch.save(modular_state_dict, '../main_models/bert_medium.pth')

print("Bert Medium Modularized")

'''

# Bert-Mini

bert_mini = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')


model_dict_bert_mini = {'l': 4, 'o': ['sa']*4, 'h': [256]*4, 'n': [4]*4, 'f': [[1024]]*4, 'p': ['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_mini)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_mini.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_mini_graph = graphLib.get_graph(model_dict=model_dict_bert_mini)[0]

if bert_mini_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-medium found in dataset!{pu.bcolors.ENDC}')
	print(bert_mini_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_mini_graph.hash)+'/')

tokenizer.save_pretrained("../models/pretrained/"+str(bert_mini_graph.hash)+'/')

print("Bert Mini Modularized")


# Bert-Tiny

bert_tiny = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

model_dict_bert_tiny= { 'l': 2, 'o': ['sa']*2, 'h': [128]*2, 'n': [2]*2, 'f': [[4*128]]*2, 'p': ['sdp']*2}

config = BertConfig()

config.from_model_dict(model_dict_bert_tiny)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_tiny.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

bert_tiny_graph = graphLib.get_graph(model_dict=model_dict_bert_tiny)[0]

if bert_tiny_graph is not None:
	print(f'{pu.bcolors.OKGREEN}BERT-tiny found in dataset!{pu.bcolors.ENDC}')
	print(bert_tiny_graph, '\n')

model_modular.save_pretrained("../models/pretrained/"+str(bert_tiny_graph.hash)+'/')

tokenizer.save_pretrained("../models/pretrained/"+str(bert_tiny_graph.hash)+'/')

print("Bert Tiny Modularized")


# Bert-L2-H256-A4

bert_2_256 = BertModel.from_pretrained('google/bert_uncased_L-2_H-256_A-4')

model_dict_bert_2_256= { 'l': 2, 'o': ['sa']*2, 'h': [256]*2, 'n': [4]*2, 'f': [[4*256]]*2, 'p': ['sdp']*2}

config = BertConfig()

config.from_model_dict(model_dict_bert_2_256)

model_modular = BertModelModular(config)

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

print("Bert L-2/H-256/A-4 Modularized")


# Bert-L4-H128-A2

bert_4_128 = BertModel.from_pretrained('google/bert_uncased_L-4_H-128_A-2')

model_dict_bert_4_128= { 'l': 4, 'o': ['sa']*4, 'h': [128]*4, 'n': [2]*4, 'f': [[4*128]]*4, 'p': ['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_4_128)

model_modular = BertModelModular(config)

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

print("Bert L-4/H-128/A-2 Modularized")

