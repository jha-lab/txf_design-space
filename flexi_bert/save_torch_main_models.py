import sys
sys.path.append('../transformers/src/')

import torch

from transformers import  BertConfig,  BertModel

from transformers.models.bert.modeling_modular_bert import BertModelModular


bert_base = BertModel.from_pretrained('bert-base-uncased')

model_dict_bert_base = {'l':12,'a':[8]*12,'f':[4*768]*12,'h':[768]*12,'s':['sdp']*12}

config = BertConfig()

config.from_model_dict(model_dict_bert_base)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_base.state_dict()

modular_state_dict.update(model_state_dict)

torch.save(modular_state_dict, '../main_models/bert_base.pth')

print("Bert Base Modularized")


bert_small = BertModel.from_pretrained('google/bert_uncased_L-4_H-512_A-8')

model_dict_bert_small = {'l':4,'a':[8]*4,'f':[4*512]*4,'h':[512]*4,'s':['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_small)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_small.state_dict()

modular_state_dict.update(model_state_dict)

torch.save(modular_state_dict, '../main_models/bert_small.pth')

print("Bert Small Modularized")


bert_medium = BertModel.from_pretrained('google/bert_uncased_L-8_H-512_A-8')

model_dict_bert_medium = {'l':8,'a':[8]*8,'f':[4*512]*8,'h':[512]*8,'s':['sdp']*8}

config = BertConfig()

config.from_model_dict(model_dict_bert_medium)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_medium.state_dict()

modular_state_dict.update(model_state_dict)

torch.save(modular_state_dict, '../main_models/bert_medium.pth')

print("Bert Medium Modularized")



bert_mini = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

model_dict_bert_mini= {'l':4,'a':[4]*4,'f':[4*256]*4,'h':[256]*4,'s':['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_mini)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_mini.state_dict()

modular_state_dict.update(model_state_dict)

torch.save(modular_state_dict, '../main_models/bert_mini.pth')

print("Bert Mini Modularized")


bert_tiny = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

model_dict_bert_tiny= {'l':2,'a':[2]*2,'f':[4*128]*2,'h':[128]*2,'s':['sdp']*2}

config = BertConfig()

config.from_model_dict(model_dict_bert_tiny)

model_modular = BertModelModular(config)

modular_state_dict = model_modular.state_dict()

model_state_dict = bert_tiny.state_dict()

modular_state_dict.update(model_state_dict)

torch.save(modular_state_dict, '../main_models/bert_tiny.pth')


print("Bert Tiny Modularized")

