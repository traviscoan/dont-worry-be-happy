# Happiness in Political Communication by US Congress Members

This repository contains the replication code and data for the paper "Happiness in political communication by men and women members of the US Congress".

## Setup Instructions

### Prerequisites
- Python 3.7+
- R 4.0+
- Required Python packages will be installed automatically
- Required R packages will be installed automatically

### Data Setup
1. Download the dataset file `data.zip` from [here](https://drive.google.com/file/d/1LWvlHNgYxSk5I-Lb3SCzjm7LLEoiNm_x/view?usp=drive_link)
2. Save the zip file in the root directory of this repository

## Running the Analysis Pipeline

The full analysis can be executed by running the provided shell script:

```bash
./run.sh
```
This script will automatically:
1. Extract the necessary data files
2. Create required output directories
3. Run all analysis steps in the correct sequence
4. Generate final results and visualizations

## Pipeline Steps

The analysis pipeline consists of the following steps:

### 1. Text Sentiment Analysis
- Processes Facebook posts from MOCs using the RoBERTa transformer model
- Generates sentiment scores (positive, negative, neutral) for each post
- Output: Sentiment analysis results stored in `output/sentiment/`

### 2. Face Emotion Detection
- Analyzes facial expressions in images of detected faces of MOCs
- Uses DeepFace to detect emotions (happiness, anger, sadness, etc.)
- Output: Emotion detection results saved in `output/emotion_processed_images/` and combined in `output/output_combined.json`

### 3. Validation Happiness Detection
- Processes validation images to extract happiness scores of detected faces in the validation set
- Output: Happiness scores stored in `output/validation/`

### 4. Validation Analysis
- Analyzes the results of the validation process
- Compares detected emotions with human-coded data
- Output: Validation metrics saved in `output/validation/`

### 5. Data Merging
- Combines the facial emotion data, text sentiment data, and metadata
- Merges image-level and post-level information
- Output: Complete dataset saved as `output/complete_face_data.csv`

### 6. Statistical Analysis
- Performs statistical modeling and hypothesis testing
- Generates tables and figures for the paper
- Output: Statistical results saved in `output/tables/` and `output/plots/`

## Output Files

The analysis generates several output files:
- `output/complete_face_data.csv`: The complete merged dataset
- `output/tables/`: Statistical tables in TeX format
- `output/plots/`: Visualizations and figures
- `output/model_observations_report.txt`: Diagnostic information on the models

## Troubleshooting

- If you encounter package installation issues, try installing them manually using `pip install` or `install.packages()` in R
- For path-related errors, ensure that `data.zip` is placed in the root directory of the repository
- If facial emotion detection is slow, consider reducing the number of images processed for testing

## Citation

If you use this code or data, please cite:

[XXXXXX]
