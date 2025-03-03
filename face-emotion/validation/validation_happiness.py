import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Get the script's directory for robust path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
validation_dir = script_dir
project_root = os.path.dirname(os.path.dirname(script_dir))

# Try multiple possible locations for data directory
possible_data_dirs = [
    os.path.join(project_root, 'data'),  # From script location
    os.path.join(os.getcwd(), 'data'),   # From current working directory
    'data',                              # Relative to current directory
    '../data',                           # One level up
    '../../data'                         # Two levels up
]

data_dir = None
for dir_path in possible_data_dirs:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        data_dir = dir_path
        print(f"Found data directory at: {data_dir}")
        break

if data_dir is None:
    raise FileNotFoundError("Could not find data directory in any expected location")

# Create directories if they don't exist
data_validation_dir = os.path.join(data_dir, 'validation_data')
os.makedirs(data_validation_dir, exist_ok=True)

output_dir = os.path.join(project_root, 'output', 'validation')
os.makedirs(output_dir, exist_ok=True)

# Find annotation file
possible_annotation_paths = [
    os.path.join(data_validation_dir, 'annotations.csv'),
    os.path.join(data_dir, 'annotations.csv'),
    os.path.join(project_root, 'data', 'validation_data', 'annotations.csv'),
]

annotation_path = None
for path in possible_annotation_paths:
    if os.path.exists(path):
        annotation_path = path
        print(f"Found annotations at: {annotation_path}")
        break

if annotation_path is None:
    raise FileNotFoundError("Could not find annotations.csv in any expected location")

# Find validation results file
possible_results_paths = [
    os.path.join(data_validation_dir, 'validation_happiness_analysis.json'),
    os.path.join(data_dir, 'validation_happiness_analysis.json'),
    os.path.join(project_root, 'data', 'validation_data', 'validation_happiness_analysis.json'),
]

results_path = None
for path in possible_results_paths:
    if os.path.exists(path):
        results_path = path
        print(f"Found validation results at: {results_path}")
        break

if results_path is None:
    raise FileNotFoundError("Could not find validation_happiness_analysis.json in any expected location")

# Read the human annotations
df = pd.read_csv(annotation_path)

# Load processed validation image data
with open(results_path, 'r') as f:
    happiness_data = json.load(f)

# Convert JSON data to DataFrame and merge
emotion_df = pd.json_normalize(happiness_data, record_path=None)
df = df.merge(emotion_df, on='image_name', how='left')

# Remove rows with NaN values
df_clean = df.dropna(subset=['happy', 'dominant_emotion'])

# Binary classification metrics for human cutoff=1 vs dominant emotion
human_binary_dom_1 = (df_clean['happy'] >= 1).astype(int)
machine_binary_dom_1 = (df_clean['dominant_emotion'] == 'happy').astype(int)

# Calculate binary metrics
accuracy_dom_1 = accuracy_score(human_binary_dom_1, machine_binary_dom_1)
precision_dom_1 = precision_score(human_binary_dom_1, machine_binary_dom_1)
recall_dom_1 = recall_score(human_binary_dom_1, machine_binary_dom_1)
f1_dom_1 = f1_score(human_binary_dom_1, machine_binary_dom_1)

print("\nBinary Classification Metrics (Human >= 1 vs Dominant Emotion = 'happy'):")
print(f"Accuracy: {accuracy_dom_1:.3f}")
print(f"Precision: {precision_dom_1:.3f}")
print(f"Recall: {recall_dom_1:.3f}")
print(f"F1 Score: {f1_dom_1:.3f}")

# Generate confusion matrix
cm_dom_1 = confusion_matrix(human_binary_dom_1, machine_binary_dom_1)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dom_1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'],
            cbar=False,
            annot_kws={'size': 14},
            )
plt.title('Confusion Matrix (Human ≥ 1 vs Dominant Emotion)', fontsize=14)
plt.ylabel('Human Label', fontsize=12)
plt.xlabel('Machine Prediction', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the confusion matrix
plt.savefig(os.path.join(output_dir, 'confusion_matrix_dominant_1.png'), bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_dominant_1.pdf'), bbox_inches='tight')
plt.close()

# Generate validation report
report_text = f"""# Happiness Detection Validation Report

## Process Overview

1. Data Preparation
   - Loaded human annotations from annotations.csv
   - Merged human annotations with machine predictions

2. Validation Dataset
   - Number of images: {len(df_clean)}
   - Scale for human annotations: 0-4 (0: not happy, 4: very happy)
   - Machine Classification: dominant_emotion == "happy"

## Validation Results

Binary Classification Metrics (Human >= 1 vs Dominant Emotion):
- Threshold for Human Binary Classification: >= 1 (on 0-4 scale)
- Machine Classification: dominant_emotion == "happy"
- Accuracy: {accuracy_dom_1:.3f}
- Precision: {precision_dom_1:.3f}
- Recall: {recall_dom_1:.3f}
- F1 Score: {f1_dom_1:.3f}
"""

# Save the report
with open(os.path.join(output_dir, 'validation_report.txt'), 'w') as f:
    f.write(report_text)

print("\nValidation report and confusion matrix have been saved.")
