# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=I+like+you.+I+love+you
# https://github.com/cardiffnlp/tweeteval

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
from tqdm import tqdm, trange
import os

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Define paths
data_dir = 'data'
cardiff_model_dir = os.path.join(data_dir, 'cardiffnlp')
output_dir = 'output/sentiment'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cardiff_model_dir, exist_ok=True)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)


'''
If you receive the following error:
OSError: Can't load tokenizer for 'cardiffnlp/twitter-roberta-base-sentiment'

Delete /cardiffnlp directory and re-install model
'''

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(os.path.join(cardiff_model_dir, f'twitter-roberta-base-{task}'))

def get_roberta_sent(dict):
    text = preprocess(dict['message'])
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sent_data =  {"roberta_negative": scores[0],
              "roberta_neutral": scores[1],
              "roberta_positive": scores[2]}
    dict.update(sent_data)
    return dict

# Load text data
print("Load text data")
facebook_data = pd.read_csv(os.path.join(data_dir, 'facebook_data.csv'), keep_default_na=False)
facebook_data = facebook_data.to_dict('records')

print("Apply Roberta model for sentiment analysis")
results = [get_roberta_sent(d) for d in tqdm(facebook_data)]

print('Write data to disk')
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, 'facebook_sentiment_roberta.csv'), index=None)

print("Finished!")