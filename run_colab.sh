#!/bin/bash

# Exit on any error
set -e

# Set up directories
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$PROJECT_DIR/output"
DRIVE_OUTPUT_DIR="/content/drive/MyDrive/dont_worry_be_happy2"

echo "Starting face emotion detection pipeline..."

# Check if Google Drive is mounted
if [ ! -d "/content/drive" ]; then
    echo "Error: Google Drive is not mounted. Please run the following in a Colab cell first:"
    echo "from google.colab import drive"
    echo "drive.mount('/content/drive')"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in the current directory."
fi

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

# Create Google Drive output directory if it doesn't exist
mkdir -p "$DRIVE_OUTPUT_DIR"

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

# Copy results to Google Drive
echo "Copying results to Google Drive..."
cp -r "$OUTPUT_DIR"/* "$DRIVE_OUTPUT_DIR/"

echo "Analysis pipeline complete! Results copied to $DRIVE_OUTPUT_DIR" 