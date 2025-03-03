#!/bin/bash

# Exit on any error
set -e

# Set up directories
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/output"

echo "Starting face emotion detection pipeline..."

# Unzip data if it exists and hasn't been unzipped yet
if [ -f data.zip ] && [ ! -d "$DATA_DIR" ]; then
    echo "Unzipping data.zip..."
    unzip data.zip
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/emotion_processed_images"
mkdir -p "$OUTPUT_DIR/sentiment"
mkdir -p "$OUTPUT_DIR/validation"

# Run text analysis
echo "Running text analysis..."
cd "$PROJECT_DIR/text-analysis"
python facebook_sentiment_transformer.py

# Run face emotion detection
echo "Running face emotion detection..."
cd "$PROJECT_DIR/face-emotion"
python face_emotion_detection.py

# Run validation happiness detection
echo "Running validation happiness detection..."
cd "$PROJECT_DIR/face-emotion/validation"
python detect_validation_happiness.py

# Run validation analysis
echo "Running validation analysis..."
cd "$PROJECT_DIR/face-emotion/validation"
python validation_happiness.py

# Run data merging
echo "Running data merging..."
cd "$PROJECT_DIR/statistical-analysis"
python merge_face_data.py

# Run statistical analysis
echo "Running statistical analysis..."
cd "$PROJECT_DIR/statistical-analysis"
Rscript statistical_analysis4.R

echo "Analysis pipeline complete!" 