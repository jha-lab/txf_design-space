import sys
sys.path.append('../transformers/src/')
sys.path.append('../embeddings/')

import torch

from transformers import  BertConfig,  ConvBertModel, BertTokenizer

from transformers.models.bert.modeling_modular_bert import BertModelModular

from library import Graph, GraphLib
from utils import print_util as pu

#Modularizing pretrained convBERT models and testing weight loading


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

conv_bert_base = ConvBertModel.from_pretrained('YituTech/conv-bert-bas')

model_dict_convbert_base = {'l':12, 't': ['c']*12, 'a':[12]*12,'f':[3072]*12,'h':[768]*12, 'nff':[1]*12,'s':['c']*12}

config = BertConfig()

config.from_model_dict(model_dict_convbert_base)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = conv_bert_base.state_dict()

modular_state_dict.update(model_state_dict)

model_modular.load_state_dict(modular_state_dict)

print("ConvBERT base modularized")

#Testing loading of weights

model_dict_random_model = {'l':3, 't': ['c','c','sa'], 'a':[12,12,8],'f':[3072,3072,1024],'h':[768,768,256], 'nff':[1,1,1],'s':['c','c','sdp']}


config = BertConfig()

config.from_model_dict(model_dict_random_model)

bert_target = BertModelModular(config)

percent = bert_target.load_model_from_source(conv_bert_base)

print("Random Model loaded with fraction:", percent)

