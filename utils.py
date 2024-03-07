''' Utility functions '''

import json
import glob
from PIL import Image, ImageDraw
import ntpath


def write_json(path, content):
    '''
    Takes a path and list of dictionaries and writes a pretty, POSIX
    compatiable JSON file.

    Args:
        path (str): Path to file where JSON should be written.
        content (list): List of dictionaries to write.
    '''

    with open(path, 'w') as f:
        json.dump(content, f, indent=4, separators=(',', ': '), sort_keys=True)
        # add trailing newline for POSIX compatibility
        f.write('\n')


def read_json(path):
    '''
    Reads a JSON file.

    Args:
        path (str): Path to JSON file to read.
    
    Returns:
        (list): List of dictionaries.
    '''
    with open(path, 'r') as f:
        content = json.load(f)
    
    return content


def read_encodings(encodings_dir):
    '''
    Read encoding JSON files from a directory.

    Args:
        encodings_dir (str): Directory of JSON files with 
                             face encodings.
    
    Returns:
        (list): List of dictionaries with the face encodings.
    '''
    encodings = []
    path_to_files = sorted(glob.glob(f'{encodings_dir}/*.json'))
    for path in path_to_files:
        encodings.append(json.load(open(path)))
    
    return encodings


def load_image(image_path):
    '''
    Load image using the PIL library.

    Args:
        image_path (str): Path to an image file.image_path
    
    Returns:
        (tuple): Returns the PIL image, the image name (str),
                 and the image extension (str).
    '''
    image_name, image_ext = ntpath.splitext(ntpath.basename(image_path))
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image, image_name, image_ext


def display_image(image, boxes=None, color=(255, 0, 0)):
    '''
    Display image with (if they exist) bounding boxes.

    Args:
        image (PIL image object): Image to display
        boxes (list of lists): Any bounding boxes detected in the image.
        color (tuple): The color (RGB) of the bounding box when displayed.
    '''

    if boxes is None:
            image.show()
    else:
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        for box in boxes:
            draw.rectangle(box, outline=color, width=6)
        image_draw.show()


def flatten(list_of_lists):
    '''
    Takes a list of lists and returns the flattned version.

    Args:
        list_of_lists (list): A list of lists to flatten
    
    Returns: 
        list: The flattened list
    '''

    return [item for sublist in list_of_lists for item in sublist]
