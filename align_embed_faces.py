'''
Detect and embed faces in images. 

This script reads the bioguide_image_level.json file and processes each image 
for each bioguide. The script uses the MTCNN detector to detect and align faces 
in each image. The aligned faces are then embedded using the FaceNet model. The 
embeddings are saved in the data/encodings directory. The aligned faces are saved 
in the data/cropped directory.
'''

import detect
import utils
import pickle
import os
import sys

def create_dir(dir):
    '''
    Create a directory if it does not exist.
    
    Args:
        dir (str): Directory to create.
    
    Returns:
        None
    '''
    if os.path.isdir(dir):
        print(f'{dir} already exists.')
    else:
        os.mkdir(dir)


if __name__ == "__main__":
    # Update Python path
    basepath = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, basepath)

    # Read the paths to all images by bioguide
    bioguide_data = utils.read_json('data/bioguide_image_level.json')
    bioguides = sorted(list(set([row['bioguide'] for row in bioguide_data])))

    # Create directory structure to save cropped images and encodings
    create_dir('data/cropped')
    create_dir('data/encodings')
    for bioguide in bioguides:
        create_dir(f'data/cropped/{bioguide}')
        create_dir(f'data/encodings/{bioguide}')
    
    # Align and embed faces
    detector = detect.MTCNNDetector()
    errors = []
    for bioguide in bioguides:
        subset = [row for row in bioguide_data if row['bioguide'] == bioguide]
        print(f'Encoding {len(subset)} images for bioguide = {bioguide}')
        encodings = []
        for i,row in enumerate(subset):
            print(f'Processing {i}, {row["image_path"]}')
            encodings.append(detector.detect_encode(row['image_path'], cropped_dir=f'data/cropped/{bioguide}'))

        # Flatten and save
        results_flat = []
        for row in encodings:
            for face in row:
                results_flat.append(face)
        
        with open(f'data/encodings/{bioguide}/encodings.pkl', 'wb') as f:
            pickle.dump(results_flat, f)
