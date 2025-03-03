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
import torch.amp
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

# Optimize tokenizer with caching for repeated texts
class TokenizerCache:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}
    
    def encode(self, texts, **kwargs):
        result = {}
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self.cache:
                result[i] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Tokenize uncached texts
        if uncached_texts:
            uncached_encodings = self.tokenizer(uncached_texts, **kwargs)
            
            # Update cache and result
            for j, i in enumerate(uncached_indices):
                encoding = {k: v[j:j+1] for k, v in uncached_encodings.items()}
                self.cache[texts[i]] = encoding
                result[i] = encoding
        
        # Combine results in original order
        combined = {k: [] for k in next(iter(result.values())).keys()}
        for i in range(len(texts)):
            for k in combined.keys():
                combined[k].append(result[i][k][0])
        
        # Convert lists to tensors
        return {k: torch.tensor(v) for k, v in combined.items()}

# Create tokenizer cache
tokenizer_cache = TokenizerCache(tokenizer)

def process_text_batch(texts, batch_size=32):
    """Process a batch of preprocessed texts and return sentiment scores"""
    all_scores = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with cache
        encoded_inputs = tokenizer_cache.encode(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        # Move to GPU
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Run inference with optimizations
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if str(device) == 'cuda' else 'cpu'):
                outputs = model(**encoded_inputs)
        
        # Process outputs
        scores = outputs[0].cpu().numpy()
        scores = softmax(scores, axis=1)
        all_scores.extend(scores)
    
    return all_scores

# Load and preprocess all data at once
print("Loading and preprocessing data...")
facebook_data = pd.read_csv(os.path.join(data_dir, 'facebook_data.csv'), keep_default_na=False, low_memory=False)
records = facebook_data.to_dict('records')

# Extract and preprocess all texts at once
all_texts = [preprocess(record['message']) for record in records]
unique_texts = list(set(all_texts))  # Remove duplicates for faster processing

print(f"Processing {len(unique_texts)} unique messages...")
start_time = time.time()
# Find optimal batch size based on GPU memory
batch_size = 64 if torch.cuda.is_available() else 32
# Process all unique texts
unique_text_scores = {}
for i in tqdm(range(0, len(unique_texts), batch_size * 10)):
    chunk = unique_texts[i:i+batch_size*10]
    scores = process_text_batch(chunk, batch_size)
    
    # Store scores in dictionary for quick lookup
    for j, text in enumerate(chunk):
        unique_text_scores[text] = {
            "roberta_negative": float(scores[j][0]),
            "roberta_neutral": float(scores[j][1]),
            "roberta_positive": float(scores[j][2])
        }
    
    # Clear cache occasionally
    if i % (batch_size * 30) == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

# Apply scores to all records
print("Applying scores to all records...")
for record in tqdm(records):
    text = preprocess(record['message'])
    scores = unique_text_scores[text]
    record.update(scores)

print(f"Processing completed in {time.time() - start_time:.2f} seconds")

print('Writing data to disk')
df = pd.DataFrame(records)
df.to_csv(os.path.join(output_dir, 'facebook_sentiment_roberta.csv'), index=None)

print("Finished!")