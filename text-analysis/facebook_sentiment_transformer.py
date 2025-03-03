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
# Use both relative and absolute path handling to make the script work from any location
import os

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to data folder from script location
data_path = os.path.join(os.path.dirname(script_dir), "data")

# Check if the file exists at this location
facebook_data_path = os.path.join(data_path, "facebook_data.csv")
if not os.path.exists(facebook_data_path):
    print(f"Could not find data at {facebook_data_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for data in current directory...")
    # Try to find data in the current working directory structure
    if os.path.exists("data/facebook_data.csv"):
        facebook_data_path = "data/facebook_data.csv"
    elif os.path.exists("../data/facebook_data.csv"):
        facebook_data_path = "../data/facebook_data.csv"
    else:
        raise FileNotFoundError(f"Could not locate facebook_data.csv in any expected location")

print(f"Loading data from: {facebook_data_path}")
rtext = pd.read_csv(facebook_data_path, keep_default_na=False)
rtext = rtext.to_dict('records')

print("Apply Roberta model for sentiment analysis")
results = [get_roberta_sent(d) for d in tqdm(rtext)]

print('Write data to disk')
df = pd.DataFrame(results)

# Create output directory using the same base location as the data file
output_base = os.path.dirname(os.path.dirname(os.path.abspath(facebook_data_path)))
output_dir = os.path.join(output_base, "output", "sentiment")
os.makedirs(output_dir, exist_ok=True)

# Save to the output file
output_file = os.path.join(output_dir, "facebook_sentiment_roberta.csv")
print(f"Saving results to: {output_file}")
df.to_csv(output_file, index=None)

print("Finished!")