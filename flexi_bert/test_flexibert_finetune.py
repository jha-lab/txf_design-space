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



bert_mini.save_pretrained("../models/")
