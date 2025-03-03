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
import torch
from torch.cuda.amp import autocast
import shutil

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Define paths
data_dir = 'data'  # For input data
cardiff_model_dir = 'cardiffnlp'  # Local to text-analysis directory
output_dir = 'output/sentiment'

# Create all necessary directories
os.makedirs(cardiff_model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# Set model cache directory to ensure models are saved in text-analysis/cardiffnlp
os.environ['TRANSFORMERS_CACHE'] = cardiff_model_dir

# Handle model loading with automatic retry
def load_tokenizer_and_model(retry=True):
    global tokenizer, model
    try:
        print(f"Loading tokenizer from {MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, cache_dir=cardiff_model_dir)
        
        print(f"Loading model from {MODEL}...")
        # PT
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=cardiff_model_dir)
        model.save_pretrained(os.path.join(cardiff_model_dir, f'twitter-roberta-base-{task}'))
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
        return True
        
    except OSError as e:
        print(f"Error loading model: {str(e)}")
        if retry:
            print(f"Attempting to fix by clearing the cache directory: {cardiff_model_dir}")
            # Remove the cache directory
            if os.path.exists(cardiff_model_dir):
                shutil.rmtree(cardiff_model_dir)
                os.makedirs(cardiff_model_dir, exist_ok=True)
            
            print("Retrying model download...")
            return load_tokenizer_and_model(retry=False)  # Retry once without allowing further retries
        else:
            print("Failed to load model even after clearing cache. Please check your internet connection.")
            raise

# Try to load the model with automatic retry
load_tokenizer_and_model()

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# Enable mixed precision for faster processing on modern GPUs
from torch.cuda.amp import autocast

def process_batch(batch_data, batch_size=32):
    """Process data in batches for much faster inference"""
    results = []
    
    for i in range(0, len(batch_data), batch_size):
        current_batch = batch_data[i:i+batch_size]
        texts = [preprocess(item['message']) for item in current_batch]
        
        # Tokenize all texts at once
        encoded_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        # Move inputs to the same device as model
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Use mixed precision for faster computation
        with autocast():
            outputs = model(**encoded_inputs)
        
        # Process all outputs at once
        scores = outputs[0].detach().cpu().numpy()
        scores = softmax(scores, axis=1)
        
        for j, item in enumerate(current_batch):
            item.update({
                "roberta_negative": float(scores[j][0]),
                "roberta_neutral": float(scores[j][1]),
                "roberta_positive": float(scores[j][2])
            })
            results.append(item)
    
    return results

# Load text data
print("Load text data")
facebook_data = pd.read_csv(os.path.join(data_dir, 'facebook_data.csv'), keep_default_na=False, low_memory=False)
facebook_data = facebook_data.to_dict('records')

print("Apply Roberta model for sentiment analysis")
# Determine optimal batch size based on available memory - adjust as needed
batch_size = 128  # Try 64, 128, or 256 depending on your GPU memory
results = []

# Process in chunks to show progress
chunk_size = 10000
for i in tqdm(range(0, len(facebook_data), chunk_size)):
    chunk = facebook_data[i:i+chunk_size]
    results.extend(process_batch(chunk, batch_size=batch_size))

print('Write data to disk')
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, 'facebook_sentiment_roberta.csv'), index=None)

print("Finished!")