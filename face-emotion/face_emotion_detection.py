# -*- coding: utf-8 -*-

import re  # Import the regular expression module
from deepface import DeepFace
import cv2
import numpy as np
import tarfile
import os
import json
import time
import tarfile

def search_file_in_tar(tar_path, target_file_name):
    # Open the tar file in read mode
    with tarfile.open(tar_path, 'r') as tar:
        # Filter members to get only files that end with '.jpg'
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]
        # Check if the target file is in the filtered list
        for member in members:
            if target_file_name in member.name:
                return True
    # Return False if the file was not found in the tar
    return False

def list_first_five_files(tar_path):
    # Open the tar file in read mode
    with tarfile.open(tar_path, 'r') as tar:
        # Filter members to get only files that end with '.jpg'
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]
        # Initialize a list to hold the names of the first five files
        file_names = [member.name for member in members[:5]]
        return file_names

"""#### Analyze images"""

tarball_path = '/content/drive/MyDrive/dont_worry_be_happy/cropped_predictions.tar.gz'
output_dir = '/content/drive/MyDrive/happiness_analysis/output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory at {output_dir}")

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

import os
import re

output_dir = '/content/drive/MyDrive/happiness_analysis/output'

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

# Load libraries to read data into colab
from google.colab import files
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

"""Change working directory:"""

cd /content/drive/MyDrive/happiness_analysis

"""Append json files to a single one for download. Uses asynchronous execution."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define the directory containing the JSON files
directory = '/content/drive/MyDrive/happiness_analysis/output/'

# Define the output JSON file name
output_filename = 'output_combined.json'

# Function to load a single JSON file
def load_json(filename):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Use ThreadPoolExecutor to parallelize file reading
with ThreadPoolExecutor() as executor:
    # Initialize a progress bar
    results = list(tqdm(executor.map(load_json, json_files), total=len(json_files)))

# Save the combined data to a single JSON file
with open(output_filename, 'w') as outfile:
    json.dump(results, outfile)

print(f'All JSON files have been combined into {output_filename}')

"""Count number of Facebook JSON files:"""

!find output/ -type f | wc -l
