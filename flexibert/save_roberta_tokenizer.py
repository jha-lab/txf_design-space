from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(tokenizer.model_max_length)
tokenizer.save_pretrained('../roberta_tokenizer/')
