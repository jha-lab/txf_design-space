import sys
sys.path.append('../transformers/src/')

import torch

from transformers import  BertConfig

from transformers.models.bert.modeling_modular_bert import BertModelModular

model_dict_bert_mini = {'l':4,'a':[4]*4,'f':[4*256]*4,'h':[256]*4,'s':['sdp']*4}

config = BertConfig()

config.from_model_dict(model_dict_bert_mini)

bert_mini = BertModelModular(config)

bert_mini.load_state_dict(torch.load('../main_models/bert_mini.pth'))

#Test 1

model_dict_target = {'l':3,'a':[4,4,8],'f':[1024,512,256],'h':[256,256,512],'s':['sdp','wma','sdp']}

config = BertConfig()

config.from_model_dict(model_dict_target)

bert_target = BertModelModular(config)

bert_target.load_model_from_source(bert_mini)


print("Model 1 loaded!")


#Test 2

model_dict_target = {'l':5,'a':[4,4,4,4,12],'f':[1024,1024,1024,512,256],'h':[256,256,256,256,512],'s':['sdp','wma','sdp','wma','sdp']}

config = BertConfig()
config.from_model_dict(model_dict_target)

bert_target = BertModelModular(config)

bert_target.load_model_from_source(bert_mini)

print("Model 2 loaded")
