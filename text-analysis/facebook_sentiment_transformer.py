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

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# Add enhanced error handling for tokenizer loading
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
except OSError:
    print("Error loading tokenizer. Attempting to resolve issues and force download...")
    import shutil
    import os
    
    # Check if there's a local "cardiffnlp" directory in current or parent directory
    current_dir = os.getcwd()
    cardiff_dir_current = os.path.join(current_dir, "cardiffnlp")
    cardiff_dir_parent = os.path.join(os.path.dirname(current_dir), "cardiffnlp")
    
    # Remove cardiffnlp directory if it exists in current directory
    if os.path.exists(cardiff_dir_current):
        print(f"Found conflicting local directory at {cardiff_dir_current}. Removing...")
        shutil.rmtree(cardiff_dir_current, ignore_errors=True)
    
    # Remove cardiffnlp directory if it exists in parent directory
    if os.path.exists(cardiff_dir_parent):
        print(f"Found conflicting local directory at {cardiff_dir_parent}. Removing...")
        shutil.rmtree(cardiff_dir_parent, ignore_errors=True)
    
    # Check if there's a local cache and remove it
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--cardiffnlp--twitter-roberta-base-sentiment")
    if os.path.exists(model_cache):
        print(f"Removing existing cache at {model_cache}")
        shutil.rmtree(model_cache, ignore_errors=True)
    
    # Force download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, force_download=True)
    print("Tokenizer successfully downloaded.")

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

def get_roberta_sent(dict):
    text = preprocess(dict['message'])
    
    # Check if text is empty or too short after preprocessing
    if not text or len(text.strip()) < 2:
        print(f"Warning: Empty or very short text after preprocessing: '{dict['message']}'")
        # Mark as invalid without assuming sentiment
        dict.update({
            "roberta_negative": 0,
            "roberta_neutral": 0,
            "roberta_positive": 0,
            "sentiment_valid": False  # Flag to identify invalid entries
        })
        return dict
    
    try:
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        sent_data = {
            "roberta_negative": scores[0],
            "roberta_neutral": scores[1],
            "roberta_positive": scores[2],
            "sentiment_valid": True
        }
        dict.update(sent_data)
        return dict
    except Exception as e:
        print(f"Error processing text: '{text}'. Error: {str(e)}")
        # Mark as invalid without assuming sentiment
        dict.update({
            "roberta_negative": 0,
            "roberta_neutral": 0,
            "roberta_positive": 0,
            "sentiment_valid": False
        })
        return dict

print("Load text data")
dir = "../data"
rtext = pd.read_csv(dir + "/facebook_data.csv", keep_default_na=False)
rtext = rtext.to_dict('records')

print("Apply Roberta model for sentiment analysis")
results = [get_roberta_sent(d) for d in tqdm(rtext)]

print('Write data to disk')
df = pd.DataFrame(results)

# Create output directory if it doesn't exist
output_dir = "../output/sentiment"
os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist

# Save to the output file
df.to_csv(f"{output_dir}/facebook_sentiment_roberta.csv", index=None)

print("Finished!")