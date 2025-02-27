import pandas as pd
import tarfile
import os
import shutil
import tempfile
import json
from sklearn.metrics import mean_squared_error
import numpy as np
import zipfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Read the CSV file from the validation folder
df = pd.read_csv('../validation/annotations.csv')

# Print the column names
print("\nColumns:")
print(df.columns.tolist())

# Print the first few rows
print("\nFirst few rows:")
print(df.head())

# Print the first row using iloc
print("\nFirst row:")
print(df.iloc[0])

# Print just the id value from the first row
print("\nID value from first row:")
print(df.iloc[0]['id'])

# Open images.zip and extract all files in images folder to validation folder

# Create a temporary directory in the validation folder
temp_dir = os.path.join(os.path.dirname(__file__), '../validation/temp_extracted')
os.makedirs(temp_dir, exist_ok=True)    

# Extract ZIP contents to temp directory
with zipfile.ZipFile('../validation/images.zip', 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Create tarball of the extracted images and save it in the validation folder
with tarfile.open('../validation/validation_images.tar.gz', 'w:gz') as tar:
    # Add each file individually at the root level
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            # Skip macOS metadata files
            if not file.startswith('._'):
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=file)

# Clean up: remove the temporary directory and its contents
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")

print("\nImages for validation have been saved. Processing happens externally. Continuing with the predicted emotions for validation set...")

# Load the JSON file
with open('../validation/validation_happiness_analysis.json', 'r') as f:
    happiness_data = json.load(f)

# Convert JSON data to DataFrame
happiness_df = pd.DataFrame(happiness_data)

# Expand the emotion dictionary into separate columns
emotion_df = pd.json_normalize(happiness_data, record_path=None)

# Merge with original DataFrame
df = df.merge(emotion_df, on='image_name', how='left')

# Print the merged result
print("\nMerged DataFrame:")
print(df.head())
print(df.columns.tolist())

# Save the merged DataFrame to CSV
df.to_csv('../validation/merged_validation_data.csv', index=False)
print("\nMerged data saved to: merged_validation_data.csv")

# Calculate Spearman correlation between human and machine happiness scores
spearman_corr = df['happy'].corr(df['emotion.happy'], method='spearman')
print("\nSpearman correlation between human happiness (0-4) and machine happiness probability (0-100):")
print(f"rho = {spearman_corr:.3f}")

# Remove rows with NaN values before calculating error metrics
df_clean = df.dropna(subset=['happy', 'emotion.happy'])

# Print summary statistics
print("\nSummary Statistics:")
print("\nHuman Happiness Scores (0-4):")
print(df_clean['happy'].describe())
print("\nMachine Happiness Predictions (0-100):")
print(df_clean['emotion.happy'].describe())

# Normalize machine scores to 0-4 scale for direct comparison
machine_normalized = df_clean['emotion.happy'] * (4/100)

# Calculate various metrics
mse = mean_squared_error(df_clean['happy'], machine_normalized)
rmse = np.sqrt(mse)
mae = np.abs(df_clean['happy'] - machine_normalized).mean()
pearson_corr = df_clean['happy'].corr(df_clean['emotion.happy'], method='pearson')

print("\nAdditional Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Pearson correlation: {pearson_corr:.3f}")

# Add binary classification metrics

# Create binary versions of human and machine annotations
human_binary = (df_clean['happy'] >= 3).astype(int)
machine_binary = (df_clean['emotion.happy'] >= 75).astype(int)  # Using 75 as threshold for 0.75 probability

# Calculate binary metrics
accuracy = accuracy_score(human_binary, machine_binary)
precision = precision_score(human_binary, machine_binary)
recall = recall_score(human_binary, machine_binary)
f1 = f1_score(human_binary, machine_binary)

print("\nBinary Classification Metrics (Human >= 3 vs Machine >= 0.75):")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Add binary classification metrics for cutoff = 2
human_binary_2 = (df_clean['happy'] >= 2).astype(int)
machine_binary_2 = (df_clean['emotion.happy'] >= 75).astype(int)  # Keeping same machine threshold

# Calculate binary metrics
accuracy_2 = accuracy_score(human_binary_2, machine_binary_2)
precision_2 = precision_score(human_binary_2, machine_binary_2)
recall_2 = recall_score(human_binary_2, machine_binary_2)
f1_2 = f1_score(human_binary_2, machine_binary_2)

print("\nBinary Classification Metrics (Human >= 2 vs Machine >= 0.75):")
print(f"Accuracy: {accuracy_2:.3f}")
print(f"Precision: {precision_2:.3f}")
print(f"Recall: {recall_2:.3f}")
print(f"F1 Score: {f1_2:.3f}")

# Add binary classification metrics for cutoff = 1
human_binary_1 = (df_clean['happy'] >= 1).astype(int)
machine_binary_1 = (df_clean['emotion.happy'] >= 75).astype(int)  # Keeping same machine threshold

# Calculate binary metrics
accuracy_1 = accuracy_score(human_binary_1, machine_binary_1)
precision_1 = precision_score(human_binary_1, machine_binary_1)
recall_1 = recall_score(human_binary_1, machine_binary_1)
f1_1 = f1_score(human_binary_1, machine_binary_1)

print("\nBinary Classification Metrics (Human >= 1 vs Machine >= 0.75):")
print(f"Accuracy: {accuracy_1:.3f}")
print(f"Precision: {precision_1:.3f}")
print(f"Recall: {recall_1:.3f}")
print(f"F1 Score: {f1_1:.3f}")

# Add confusion matrix visualization

# Calculate confusion matrix
cm = confusion_matrix(human_binary, machine_binary)

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'])
plt.title('Confusion Matrix: Human vs Machine Happiness Detection')
plt.ylabel('Human Annotation')
plt.xlabel('Machine Prediction')

# Save the confusion matrix plot
plt.tight_layout()
plt.savefig('../validation/confusion_matrix.png')
plt.close()

# Generate confusion matrix for cutoff=2
cm_2 = confusion_matrix(human_binary_2, machine_binary_2)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_2, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'],
            cbar=False,  # Remove the color bar/legend
            annot_kws={'size': 14},  # Larger numbers in cells
            )
plt.title('Confusion Matrix (Human cutoff >= 2)', fontsize=14)
plt.ylabel('Human Label', fontsize=12)
plt.xlabel('Machine Prediction', fontsize=12)
plt.xticks(fontsize=12)  # Larger x-axis labels
plt.yticks(fontsize=12)  # Larger y-axis labels

# Save the confusion matrix in both formats
plt.savefig('../validation/confusion_matrix_cutoff_2.png', bbox_inches='tight')
plt.savefig('../validation/confusion_matrix_cutoff_2.pdf', bbox_inches='tight')
plt.close()

print("\nConfusion matrix for cutoff=2 has been saved as 'confusion_matrix_cutoff_2.png' and 'confusion_matrix_cutoff_2.pdf'")

# Generate confusion matrix for cutoff=1
cm_1 = confusion_matrix(human_binary_1, machine_binary_1)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'],
            cbar=False,  # Remove the color bar/legend
            annot_kws={'size': 14},  # Larger numbers in cells
            )
plt.title('Confusion Matrix (Human cutoff >= 1)', fontsize=14)
plt.ylabel('Human Label', fontsize=12)
plt.xlabel('Machine Prediction', fontsize=12)
plt.xticks(fontsize=12)  # Larger x-axis labels
plt.yticks(fontsize=12)  # Larger y-axis labels

# Save the confusion matrix in both formats
plt.savefig('../validation/confusion_matrix_cutoff_1.png', bbox_inches='tight')
plt.savefig('../validation/confusion_matrix_cutoff_1.pdf', bbox_inches='tight')
plt.close()

print("\nConfusion matrix for cutoff=1 has been saved as 'confusion_matrix_cutoff_1.png' and 'confusion_matrix_cutoff_1.pdf'")

# Add ROC and PR curve analysis

# Create binary version of human annotations but keep machine continuous
human_binary = (df_clean['happy'] >= 3).astype(int)
machine_continuous = df_clean['emotion.happy'] / 100  # Normalize to 0-1

# Calculate ROC AUC and Average Precision
roc_auc = roc_auc_score(human_binary, machine_continuous)
avg_precision = average_precision_score(human_binary, machine_continuous)

print("\nContinuous Machine vs Binary Human Metrics:")
print(f"ROC AUC Score: {roc_auc:.3f}")
print(f"Average Precision Score: {avg_precision:.3f}")

# Add binary classification metrics for human cutoff=2 vs dominant emotion
human_binary_dom = (df_clean['happy'] >= 2).astype(int)
machine_binary_dom = (df_clean['dominant_emotion'] == 'happy').astype(int)

# Calculate binary metrics
accuracy_dom = accuracy_score(human_binary_dom, machine_binary_dom)
precision_dom = precision_score(human_binary_dom, machine_binary_dom)
recall_dom = recall_score(human_binary_dom, machine_binary_dom)
f1_dom = f1_score(human_binary_dom, machine_binary_dom)

print("\nBinary Classification Metrics (Human >= 2 vs Dominant Emotion = 'happy'):")
print(f"Accuracy: {accuracy_dom:.3f}")
print(f"Precision: {precision_dom:.3f}")
print(f"Recall: {recall_dom:.3f}")
print(f"F1 Score: {f1_dom:.3f}")

# Generate confusion matrix for dominant emotion analysis
cm_dom = confusion_matrix(human_binary_dom, machine_binary_dom)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dom, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'],
            cbar=False,  # Remove the color bar/legend
            annot_kws={'size': 14},  # Larger numbers in cells
            )
plt.title('Confusion Matrix (Human ≥ 2 vs Dominant Emotion)', fontsize=14)
plt.ylabel('Human Label', fontsize=12)
plt.xlabel('Machine Prediction', fontsize=12)
plt.xticks(fontsize=12)  # Larger x-axis labels
plt.yticks(fontsize=12)  # Larger y-axis labels

# Save the confusion matrix in both formats
plt.savefig('../validation/confusion_matrix_dominant.png', bbox_inches='tight')
plt.savefig('../validation/confusion_matrix_dominant.pdf', bbox_inches='tight')
plt.close()

print("\nConfusion matrix for dominant emotion analysis has been saved as 'confusion_matrix_dominant.png' and 'confusion_matrix_dominant.pdf'")

# Add binary classification metrics for human cutoff=1 vs dominant emotion
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

# Generate confusion matrix for dominant emotion analysis (cutoff=1)
cm_dom_1 = confusion_matrix(human_binary_dom_1, machine_binary_dom_1)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dom_1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Happy', 'Happy'],
            yticklabels=['Not Happy', 'Happy'],
            cbar=False,  # Remove the color bar/legend
            annot_kws={'size': 14},  # Larger numbers in cells
            )
plt.title('Confusion Matrix (Human ≥ 1 vs Dominant Emotion)', fontsize=14)
plt.ylabel('Human Label', fontsize=12)
plt.xlabel('Machine Prediction', fontsize=12)
plt.xticks(fontsize=12)  # Larger x-axis labels
plt.yticks(fontsize=12)  # Larger y-axis labels

# Save the confusion matrix in both formats
plt.savefig('../validation/confusion_matrix_dominant_1.png', bbox_inches='tight')
plt.savefig('../validation/confusion_matrix_dominant_1.pdf', bbox_inches='tight')
plt.close()

print("\nConfusion matrix for dominant emotion analysis (cutoff=1) has been saved as 'confusion_matrix_dominant_1.png' and 'confusion_matrix_dominant_1.pdf'")

# Generate validation report
report_text = f"""# Happiness Detection Validation Report

## Process Overview

1. Data Preparation
   - Loaded human annotations from annotations.csv
   - Extracted validation images from images.zip
   - Created validation_images.tar.gz for external processing
   - Processed images through happiness detection model
   - Merged human annotations with machine predictions

2. Validation Dataset
   - Number of images: {len(df_clean)}
   - Scale for human annotations: 0-4 (0: not happy, 4: very happy)
   - Scale for machine predictions: 0-100 (probability of happiness)

## Validation Results

Continuous Metrics (Machine scores normalized to 0-4 scale):
- Mean Squared Error (MSE): {mse:.3f}
- Root Mean Squared Error (RMSE): {rmse:.3f}
- Mean Absolute Error (MAE): {mae:.3f}
- Pearson correlation: {pearson_corr:.3f}
- Spearman correlation: {spearman_corr:.3f}

Binary Classification Metrics (Threshold-based, Cutoff = 3):
- Threshold for Human Binary Classification: >= 3 (on 0-4 scale)
- Threshold for Machine Binary Classification: >= 75 (on 0-100 scale)
- Accuracy: {accuracy:.3f}
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1 Score: {f1:.3f}

Binary Classification Metrics (Threshold-based, Cutoff = 2):
- Threshold for Human Binary Classification: >= 2 (on 0-4 scale)
- Threshold for Machine Binary Classification: >= 75 (on 0-100 scale)
- Accuracy: {accuracy_2:.3f}
- Precision: {precision_2:.3f}
- Recall: {recall_2:.3f}
- F1 Score: {f1_2:.3f}

Binary Classification Metrics (Threshold-based, Cutoff = 1):
- Threshold for Human Binary Classification: >= 1 (on 0-4 scale)
- Threshold for Machine Binary Classification: >= 75 (on 0-100 scale)
- Accuracy: {accuracy_1:.3f}
- Precision: {precision_1:.3f}
- Recall: {recall_1:.3f}
- F1 Score: {f1_1:.3f}

Binary Classification Metrics (Human >= 2 vs Dominant Emotion):
- Threshold for Human Binary Classification: >= 2 (on 0-4 scale)
- Machine Classification: dominant_emotion == "happy"
- Accuracy: {accuracy_dom:.3f}
- Precision: {precision_dom:.3f}
- Recall: {recall_dom:.3f}
- F1 Score: {f1_dom:.3f}

Binary Classification Metrics (Human >= 1 vs Dominant Emotion):
- Threshold for Human Binary Classification: >= 1 (on 0-4 scale)
- Machine Classification: dominant_emotion == "happy"
- Accuracy: {accuracy_dom_1:.3f}
- Precision: {precision_dom_1:.3f}
- Recall: {recall_dom_1:.3f}
- F1 Score: {f1_dom_1:.3f}

A confusion matrix visualization has been saved as 'confusion_matrix.png' showing the distribution
of true/false positives and negatives. The matrix shows:
- True Negatives (top-left): {cm[0,0]} cases
- False Positives (top-right): {cm[0,1]} cases
- False Negatives (bottom-left): {cm[1,0]} cases
- True Positives (bottom-right): {cm[1,1]} cases

Binary Classification Metrics (Continuous Machine vs Binary Human):
- ROC AUC Score: {roc_auc:.3f}
- Average Precision Score: {avg_precision:.3f}

Summary Statistics:

Human Happiness Scores (0-4):
{df_clean['happy'].describe().to_string()}

Machine Happiness Predictions (0-100):
{df_clean['emotion.happy'].describe().to_string()}

Interpretation:
The validation results show the performance of our happiness detection model compared to human annotations. 
The Spearman correlation of {spearman_corr:.3f} indicates a {"strong" if spearman_corr > 0.7 else "moderate" if spearman_corr > 0.4 else "weak"} relationship between human and machine ratings.

For binary classification (happy vs. not happy), the model achieves an F1 score of {f1:.3f}, 
with precision at {precision:.3f} and recall at {recall:.3f}. This suggests the model {"performs well" if f1 > 0.7 else "shows moderate performance" if f1 > 0.5 else "needs improvement"} 
at distinguishing between happy and not happy expressions.

The continuous metrics show a Root Mean Squared Error of {rmse:.3f} on the 0-4 scale, 
indicating the average magnitude of prediction errors.

The ROC AUC score of {roc_auc:.3f} indicates {"excellent" if roc_auc > 0.9 else "good" if roc_auc > 0.8 else "fair" if roc_auc > 0.7 else "poor"} 
discriminative ability of the continuous machine predictions for binary happiness classification. 
The Average Precision score of {avg_precision:.3f} shows the model's precision-recall trade-off performance.
"""

# Save the report
with open('../validation/validation_report.txt', 'w') as f:
    f.write(report_text)

print("\nValidation report and confusion matrix have been saved.")
