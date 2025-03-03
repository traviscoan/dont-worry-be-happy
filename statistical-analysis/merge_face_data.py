import json
import pandas as pd
import os

# Get the script's directory for robust path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Try multiple possible locations for data directory
possible_data_dirs = [
    os.path.join(project_root, 'data'),  # From script location
    os.path.join(os.getcwd(), 'data'),   # From current working directory
    'data',                              # Relative to current directory
    '../data'                            # One level up
]

data_dir = None
for dir_path in possible_data_dirs:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        data_dir = dir_path
        print(f"Found data directory at: {data_dir}")
        break

if data_dir is None:
    raise FileNotFoundError("Could not find data directory in any expected location")

# Define output directory
output_dir = os.path.join(project_root, 'output')
sentiment_dir = os.path.join(output_dir, 'sentiment')

# Load JSON files
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find required file: {file_path}")
    
    print(f"Loading: {file_path}")
    with open(file_path, 'r') as file:
        return json.load(file)

# Define all possible paths for each required file
possible_recognition_paths = [
    os.path.join(data_dir, 'recognition_predictions.json'),
    os.path.join(project_root, 'data', 'recognition_predictions.json')
]

possible_output_combined_paths = [
    os.path.join(output_dir, 'output_combined.json'),
    os.path.join(project_root, 'output', 'output_combined.json')
]

possible_facebook_content_paths = [
    os.path.join(data_dir, 'facebook_content_image_level.json'),
    os.path.join(project_root, 'data', 'facebook_content_image_level.json')
]

possible_facebook_sentiment_paths = [
    os.path.join(sentiment_dir, 'facebook_sentiment_roberta.json'),
    os.path.join(output_dir, 'sentiment', 'facebook_sentiment_roberta.json'),
    os.path.join(project_root, 'output', 'sentiment', 'facebook_sentiment_roberta.json')
]

# Find each file
def find_file(possible_paths, file_desc):
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found {file_desc} at: {path}")
            return path
    raise FileNotFoundError(f"Could not find {file_desc} in any expected location")

# Load each required file
recognition_path = find_file(possible_recognition_paths, "recognition predictions")
output_combined_path = find_file(possible_output_combined_paths, "output combined")
facebook_content_path = find_file(possible_facebook_content_paths, "facebook content")
facebook_sentiment_path = find_file(possible_facebook_sentiment_paths, "facebook sentiment")

recognition_predictions = load_json(recognition_path)
output_combined = load_json(output_combined_path)
facebook_content_image_level = load_json(facebook_content_path)
facebook_sentiment = load_json(facebook_sentiment_path)

# Filter function for recognition_predictions
def is_facebook_and_high_score(item):
    is_facebook = all(c.isdigit() or c == '_' for c in item['face_id'])  # Check if face_id is Facebook image
    is_high_score = item['score'] >= 0.7  # Check if score is >= threshold
    return is_facebook and is_high_score

filtered_recognition_predictions = [item for item in recognition_predictions if is_facebook_and_high_score(item)]

# Convert to DataFrames
df_recognition_predictions = pd.DataFrame(filtered_recognition_predictions)
df_output_combined = pd.DataFrame(output_combined)

# Merge on face_id
merged_df = pd.merge(df_recognition_predictions, df_output_combined, on='face_id', how='left')

# Split 'emotion' column into separate columns
emotion_df = pd.json_normalize(merged_df['emotion'])
# Concatenate the new emotion columns back to the merged_df
merged_df = pd.concat([merged_df.drop('emotion', axis=1), emotion_df], axis=1)

# Prepare facebook_content_image_level DataFrame
df_facebook_content = pd.DataFrame(facebook_content_image_level)
df_facebook_content['image_name'] = df_facebook_content['filename'].str.replace('.jpg', '', regex=False)
# Drop duplicates
df_facebook_content = df_facebook_content.drop_duplicates(subset=['image_name'], keep='first')

# Merge with merged_df on image_name
merged_df2 = pd.merge(merged_df, df_facebook_content, on='image_name', how='left')

# Prepare facebook_sentiment DataFrame
df_facebook_sentiment = pd.DataFrame(facebook_sentiment)
# keep only post_url, roberta_negative, roberta_neutral, roberta_positive
df_facebook_sentiment = df_facebook_sentiment[['post_url', 'roberta_negative', 'roberta_neutral', 'roberta_positive']]
# Drop duplicates by post_url
df_facebook_sentiment = df_facebook_sentiment.drop_duplicates(subset=['post_url'], keep='first')
# Merge with merged_df2 on post_url
final_df = pd.merge(merged_df2, df_facebook_sentiment, on='post_url', how='left')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_csv_path = os.path.join(output_dir, 'complete_face_data.csv')
print(f"Saving merged data to: {output_csv_path}")
final_df.to_csv(output_csv_path, index=False)

print("Data merge completed successfully!")

