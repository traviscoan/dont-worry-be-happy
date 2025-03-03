# -*- coding: utf-8 -*-
import tarfile
import os
import json
import time
import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

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

# Create validation data directory
data_validation_dir = os.path.join(data_dir, 'validation_data')
os.makedirs(data_validation_dir, exist_ok=True)

# Define paths
possible_tarball_paths = [
    os.path.join(data_validation_dir, 'validation_images.tar.gz'),
    os.path.join(data_dir, 'validation_images.tar.gz'),
    os.path.join(project_root, 'data', 'validation_data', 'validation_images.tar.gz'),
    os.path.join(os.getcwd(), 'data', 'validation_data', 'validation_images.tar.gz')
]

tarball_path = None
for path in possible_tarball_paths:
    if os.path.exists(path):
        tarball_path = path
        print(f"Found validation images tarball at: {tarball_path}")
        break

if tarball_path is None:
    raise FileNotFoundError("Could not find validation_images.tar.gz in any expected location")

output_json_path = os.path.join(data_validation_dir, 'validation_happiness_analysis.json')

# Initialize list to store results
emotion_results = []

# Function to read image from tarball
def read_image_from_tar(tar, member):
    file_content = tar.extractfile(member).read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    return image

# Open tar file and process images
print(f"Processing tarball: {tarball_path}")

with tarfile.open(tarball_path, "r:gz") as tar:
    members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]
    total_members = len(members)
    print(f"Found {total_members} images to process.")

    start_time = time.time()  # Start timing

    for member in tqdm(members, desc="Processing images", unit="img"):
        image_name = os.path.basename(member.name)  # Extract filename only

        try:
            img = read_image_from_tar(tar, member)
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)[0]

            if isinstance(result, dict) and 'emotion' in result and 'dominant_emotion' in result:
                emotion_results.append({
                    'image_name': image_name,  # Store filename
                    'emotion': result['emotion'],
                    'dominant_emotion': result['dominant_emotion']
                })
            else:
                print(f"Unexpected result structure for {image_name}, skipping.")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    end_time = time.time()  # End timing

# Save all results to a single JSON file at the end
with open(output_json_path, 'w') as f:
    json.dump(emotion_results, f, indent=4)

print(f"Processing complete. Total processed: {len(emotion_results)}. Time taken: {end_time - start_time:.2f} seconds.")
print(f"Emotion analysis saved to {output_json_path}")

