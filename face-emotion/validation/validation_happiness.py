import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Create directories if they don't exist
output_dir = 'output/validation'
os.makedirs(output_dir, exist_ok=True)

data_validation_dir = 'data/validation_data'
os.makedirs(data_validation_dir, exist_ok=True)

# Read the human annotations
df = pd.read_csv('data/validation_data/annotations.csv')

# Load processed validation image data
with open('data/validation_data/validation_happiness_analysis.json', 'r') as f:
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
