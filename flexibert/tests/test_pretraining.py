import sys
sys.path.append('../../transformers/src/')

from transformers import  BertConfig,  BertTokenizer
from transformers.models.bert.modeling_modular_bert import BertModelModular


import torch

model_dict = {'l':3,'a':[2,4,8],'f':[512,1024,4096],'h':[512,1024,256],'s':['sdp','wma','sdp']}

config = BertConfig()

config.from_model_dict(model_dict)

model = BertModelModular(config)
print(model.config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors = "pt")
outputs = model(**inputs)
print("Output embeddings shape:",outputs[0].shape[-1])
