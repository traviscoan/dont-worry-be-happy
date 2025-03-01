# -*- coding: utf-8 -*-
import tarfile
import os
import json
import time
import cv2
import numpy as np
from deepface import DeepFace

# Create directory if it doesn't exist
data_validation_dir = 'data/validation_data'
os.makedirs(data_validation_dir, exist_ok=True)

# Define paths
tarball_path = os.path.join(data_validation_dir, 'validation_images.tar.gz')
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

    for member in members:
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
                print(f"Processed {image_name} successfully.")
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

