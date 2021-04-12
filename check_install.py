# Script to check installation of huggingface/transformers repo
# Author : Shikhar Tuli

import sys
sys.path.append('./transformers/src/')

from transformers import pipeline

print('-'*5, 'Checking installation of transformers repo', '-'*5)

nlp = pipeline("sentiment-analysis")

query = 'I love transformers!'

print(f'Input query for sentiment-analysis task: {query}')

result = nlp(query)[0]
print(f"Output label: {result['label']}, with score: {round(result['score'], 4)}")
