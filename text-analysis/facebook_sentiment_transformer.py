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
from tqdm import tqdm
import os
import torch
import shutil
import time
import gc

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

# Define device globally so it can be accessed from other functions
# Switch to CPU if we're having persistent GPU memory issues
use_gpu = True  # Set to False to force CPU usage if needed
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

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

# Simple tokenizer cache
tokenizer_cache = {}

def process_unique_messages(unique_messages, batch_size=8):
    """Process a list of unique messages and return a dictionary of results"""
    results_dict = {}
    
    for i in tqdm(range(0, len(unique_messages), batch_size)):
        batch_texts = unique_messages[i:i+batch_size]
        batch_encodings = []
        
        # Tokenize texts, using cache where possible
        for text in batch_texts:
            if text in tokenizer_cache:
                encoded = tokenizer_cache[text]
            else:
                encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
                tokenizer_cache[text] = encoded
            batch_encodings.append(encoded)
        
        # Process each text individually (avoids padding issues)
        batch_scores = []
        for encoded in batch_encodings:
            # Move inputs to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**encoded)
            
            # Get scores
            scores = outputs[0].detach().cpu().numpy()
            scores = softmax(scores, axis=1)[0]  # Get first (only) item
            batch_scores.append(scores)
        
        # Store results
        for j, text in enumerate(batch_texts):
            results_dict[text] = {
                "roberta_negative": float(batch_scores[j][0]),
                "roberta_neutral": float(batch_scores[j][1]),
                "roberta_positive": float(batch_scores[j][2])
            }
        
        # Clear GPU cache occasionally
        if i % (batch_size * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results_dict

# Load and preprocess data
print("Loading data...")
facebook_data = pd.read_csv(os.path.join(data_dir, 'facebook_data.csv'), keep_default_na=False, low_memory=False, nrows=100)
records = facebook_data.to_dict('records')

# Extract unique messages for efficient processing
print("Extracting unique messages...")
all_messages = [record['message'] for record in records]
unique_messages = list(set(all_messages))
print(f"Found {len(unique_messages)} unique messages out of {len(all_messages)} total")

# Process unique messages
print("Processing unique messages...")
start_time = time.time()
unique_results = process_unique_messages(unique_messages, batch_size=8)
print(f"Processing completed in {time.time() - start_time:.2f} seconds")

# Apply results to all records
print("Applying results to all records...")
for record in tqdm(records):
    message = record['message']
    preprocessed = preprocess(message)
    scores = unique_results.get(preprocessed, {
        "roberta_negative": 0.0,
        "roberta_neutral": 0.0,
        "roberta_positive": 0.0
    })
    record.update(scores)

# Save results
print("Saving results...")
output_df = pd.DataFrame(records)
output_df.to_csv(os.path.join(output_dir, 'facebook_sentiment_roberta.csv'), index=False)

print("Finished!")