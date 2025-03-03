#!/bin/bash

# Exit on any error
set -e

echo "Starting face emotion detection pipeline..."

# Check if data.zip exists
if [ ! -f "data.zip" ]; then
    echo "Error: data.zip not found in current directory"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Unzip data.zip
echo "Unzipping data.zip..."
unzip -o data.zip 

# Run text analysis script
#echo "Running text analysis..."
#python text-analysis/facebook_sentiment_transformer.py

# Run face emotion detection script
echo "Running face emotion detection..."
python face-emotion/face_emotion_detection.py

echo "Pipeline completed successfully!" 