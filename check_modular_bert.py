import sys
sys.path.append('./transformers/src/')

from transformers import BertTokenizer,BertLMHeadModelModular, BertConfig

import torch

model_dict = {'l':3,'a':[10,12,14],'f':[512,1024,4096],'h':[512,1024,256],'s':['sdp','wma','sdp']}

config = BertConfig()

config.from_model_dict(model_dict)

model = BertLMHeadModelModular(config)

inputs = tokenizer("Hello, my dog is cute", return_tensors = "pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = output.seq_relationship_logits
