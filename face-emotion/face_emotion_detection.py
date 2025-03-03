# -*- coding: utf-8 -*-

import re  # Import the regular expression module
from deepface import DeepFace
import cv2
import numpy as np
import tarfile
import os
import re
import json
import time
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

"""#### Analyze images"""

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

# Create output directory paths relative to project root
output_dir = os.path.join(project_root, 'output', 'emotion_processed_images')
os.makedirs(output_dir, exist_ok=True)

# Define paths
validation_data_dir = os.path.join(data_dir, 'validation_data')
tarball_path = os.path.join(validation_data_dir, 'cropped_predictions_facebook.tar.gz')
output_combined = os.path.join(project_root, 'output', 'output_combined.json')

# Check multiple possible locations for the tarball if not found
if not os.path.exists(tarball_path):
    print(f"Tarball not found at: {tarball_path}")
    print("Searching for tarball in alternative locations...")
    
    alternative_paths = [
        os.path.join(data_dir, 'cropped_predictions_facebook.tar.gz'),
        os.path.join(project_root, 'data', 'validation_data', 'cropped_predictions_facebook.tar.gz'),
        os.path.join(os.getcwd(), 'data', 'validation_data', 'cropped_predictions_facebook.tar.gz'),
        'data/validation_data/cropped_predictions_facebook.tar.gz',
        '../data/validation_data/cropped_predictions_facebook.tar.gz'
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            tarball_path = alt_path
            print(f"Found tarball at alternative location: {tarball_path}")
            break

# Final validation
if not os.path.exists(tarball_path):
    raise FileNotFoundError(f"Tarball not found in any expected location. Please ensure 'cropped_predictions_facebook.tar.gz' exists in the data directory.")

print(f"Processing tarball at {tarball_path}")

def read_image_from_tar(tar, member):
    file_content = tar.extractfile(member).read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    return image

processed_files = 0
error_files = 0
unexpected_results = 0

# Define a function to check if the filename indicates a Facebook image
def is_facebook_image(filename):
    # Facebook images seem to follow a numeric pattern separated by underscores
    # No need to split by '/' as filenames are directly provided without directory paths
    return re.match(r'^\d+(_\d+)+$', filename)

with tarfile.open(tarball_path, "r:gz") as tar:
    members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]
    total_members = len(members)
    print(f"Found {total_members} image files to process.")

    # Start processing time measurement
    start_time = time.time()

    for member in members:  # Adjust this as needed for full processing
        face_id = os.path.splitext(os.path.basename(member.name))[0]

        # Check if the image is from Facebook
        if not is_facebook_image(face_id):
            print(f"Skipping {face_id}, not a Facebook image.")
            continue

        user_bioguide = member.name.split('/')[1]
        output_file = os.path.join(output_dir, f'{face_id}.json')

        if os.path.exists(output_file):
            print(f"Skipping {face_id}, analysis already exists.")
            continue

        try:
            img = read_image_from_tar(tar, member)
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)[0]

            if isinstance(result, dict) and 'emotion' in result and 'dominant_emotion' in result:
                emotion_data = {
                    'face_id': face_id,
                    'user_bioguide': user_bioguide,
                    'emotion': result['emotion'],
                    'dominant_emotion': result['dominant_emotion']
                }

                with open(output_file, 'w') as f:
                    json.dump(emotion_data, f)

                processed_files += 1
                print(f"Successfully processed {face_id}.")
            else:
                unexpected_results += 1
                print(f"Unexpected result structure for {face_id}, skipping.")

        except Exception as e:
            error_files += 1
            print(f"Error processing {face_id}: {e}")

    # End processing time measurement
    end_time = time.time()

# Calculate and print the processing time
processing_time = end_time - start_time
print(f"Processing complete. Total files processed: {processed_files}. Unexpected results: {unexpected_results}. Errors encountered: {error_files}.")
print(f"Total processing time for Facebook images: {processing_time} seconds.")


def is_twitter_image(filename):
    # Twitter images are identified by having at least one alphabetical character in the filename
    return re.search(r'[a-zA-Z]', filename)

# List all files in the output directory
files = os.listdir(output_dir)

# Filter and delete files corresponding to Twitter images
for file in files:
    if file.endswith('.json'):  # Ensure we are only looking at JSON files
        # Extract face_id (filename without the .json extension)
        face_id = os.path.splitext(file)[0]

        # Check if the file corresponds to a Twitter image
        if is_twitter_image(face_id):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)  # Delete the file
            print(f"Deleted Twitter image JSON: {file}")

"""### Delete Twitter output from output directory"""

def is_twitter_image(filename):
    # Twitter images are identified by having at least one alphabetical character in the filename
    return re.search(r'[a-zA-Z]', filename)

# List all files in the output directory
files = os.listdir(output_dir)

# Filter and delete files corresponding to Twitter images
for file in files:
    if file.endswith('.json'):  # Ensure we are only looking at JSON files
        # Extract face_id (filename without the .json extension)
        face_id = os.path.splitext(file)[0]

        # Check if the file corresponds to a Twitter image
        if is_twitter_image(face_id):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)  # Delete the file
            print(f"Deleted Twitter image JSON: {file}")

"""## Data management"""

"""Append json files to a single one for download. Uses asynchronous execution."""

# Function to load a single JSON file
def load_json(filename):
    # Construct the full file path
    file_path = os.path.join(output_dir, filename)
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

# Use ThreadPoolExecutor to parallelize file reading
with ThreadPoolExecutor() as executor:
    # Initialize a progress bar
    results = list(tqdm(executor.map(load_json, json_files), total=len(json_files)))

# Save the combined data to a single JSON file
with open(output_combined, 'w') as outfile:
    json.dump(results, outfile)

print(f'All JSON files have been combined into {output_combined}')
