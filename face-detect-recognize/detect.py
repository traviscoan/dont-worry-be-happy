"""
Align and encode faces in an image using MTCNN and FaceNet.

Example usage:

# Instantiate detector
detector = RetinaDetector()

# Give the path to an image on your machine
image_path = "[PATH TO IMAGE ON DISK]"

# Either detect and encode all of the faces an image:
results = detector.detect_encode(image_path)

# Or you can manually do this by calling the underlying methods:
boxes,probs,landmarks = detector.detect(image_path)
encodings = detector.detect_encode(image_path)

"""

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from retinaface.pre_trained_models import get_model
import utils
import warnings
warnings.filterwarnings("ignore")

# Use GPU if you have one
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class MTCNNDetector:
    def __init__(self, mtcnn_thresholds = [0.6, 0.7, 0.7], confidence_threshold=.95, size_threshold = 20, facenet_model='vggface2', margin=0):
        '''
        Class to detect faces using MTCNN and encode the faces using FaceNet.

        Args:
            mtcnn_thresholds (list, optional): List of thresholds to use
                            for each stage of the MTCNN algorithm.
            confidence_threshold (float): Only return faces with probabilities over
                            this threshold.
            size_threshold (int): Only return faces where the smallest dimension
                            (height of width in pixels) meets this threshold.
            facenet_model (str, optional): The pretrained model to use for encoding.
            margin (int): Add a margin to the detected bounding box (in pixels).
        '''
        self.mtcnn = MTCNN(select_largest=False, keep_all=True, thresholds=mtcnn_thresholds, device=DEVICE, margin=margin)
        self.device = DEVICE
        self.facenet = InceptionResnetV1(pretrained=facenet_model).eval()
        self.confidence_threshold = confidence_threshold
        self.size_threshold = size_threshold
        # Preallocate class attributes
        self.image = None
        self.aligned = None

    def detect(self, image_path, cropped_dir=None):
        '''
        Uses an instantiated MTCNN model to detect the bounding boxes 
        and probabilities for each face.

        Args:
            image_path (str): Path to an image file.
            cropped_dir (str, optional): Directory for saving cropped faces. 
                Default is None.
        
        Returns:
            tuple(boxes, probs): A tuple of bounding box coordinates and 
                                 probabilities. Ordered from most to least
                                 probable. Boxes and probs are NumPy arrays.
                                 Each bounding box array is organized as
                                 [x0, y0, x1, y2], where (x0, y0) represents
                                 the top left point and (x1, y1) represents
                                 the bottom right point.
        '''
        # Load image using the PIL library and extract useful
        # meta data on the image to save
        self.image, self.image_name, self.image_ext = utils.load_image(image_path)
        # Extract bounding boxes, probabilities, and landmarks
        boxes,probs,landmarks = self.mtcnn.detect(self.image, landmarks=True)
        faces = []
        # If there are no faces detect, return empty lists
        if boxes is not None:
            # Filter detected faces for "quality" (min prob and min size)
            for i,bbox in enumerate(boxes):
                # Check if face meets probability threshold
                if probs[i] >= self.confidence_threshold:
                    cropped_image = self.image.crop(bbox)
                    # Check if all dimensions of cropped face are greater than
                    # the minimum size threshold
                    if all(x > self.size_threshold for x in cropped_image.size):
                        faces.append({
                            'index': i,
                            'bbox': bbox.tolist(),
                            'prob': probs[i],
                            'landmarks': landmarks[i].tolist(),
                        })
                        # Save a cropped image of the face?
                        if cropped_dir is not None:
                            cropped_image.save(f'{cropped_dir}/{self.image_name}_{i}{self.image_ext}')
            # Extract face tensors
            aligned = self.mtcnn(self.image)
            # Keep faces that pass "quality" thresholds
            idx = [face['index'] for face in faces]
            self.aligned = aligned[idx,:,:,:]
            # If no faces pass threshold, set aligned to None
            if self.aligned.nelement() == 0:
                self.aligned = None
        return faces
    
    def encode(self):
        '''
        Encodes detected faces using the FaceNet model.

        Returns:
            list: List of encodings or None if no faces are detected.
        '''
        if self.image == None:
            raise Exception('No image detected. Please run MTCNNDetector.detect() and try again.')

        # Encode
        if self.aligned is not None:
            # Prepare (stack) multiple faces for encoding.
            faces = torch.stack([face for face in self.aligned])
            return self.facenet(faces).detach().tolist()
    
    def detect_encode(self, image_path, cropped_dir=None):
        '''
        Detects and encodes an image on disk, and returns a dictionary
        of the results.
        
        Args:
            image_path (str): Path to an image file.
            cropped_dir (str, optional): Directory for saving cropped faces. 
                Default is None.
        
        Returns:
            dict: Dictionary with the detection/encoding results for
                  the image.
        '''
        faces = self.detect(image_path, cropped_dir)
        encodings = self.encode()
        faces_encodings = []
        if len(faces) == 0:
            faces_encodings = None
        else:
            if len(faces) == len(encodings):
                for i,face in enumerate(faces):
                    faces_encodings.append({
                            'face_id': f'{self.image_name}_{i}',
                            'image_name': self.image_name,
                            'image_path': image_path,
                            'bbox': face['bbox'],
                            'prob': face['prob'],
                            'landmarks': face['landmarks'],
                            'encodings': encodings[i]
                        })
        return faces_encodings


class RetinaDetector:
    def __init__(self, confidence_threshold=.90, facenet_model='vggface2'):
        self.retina = get_model("resnet50_2020-07-20", max_size=2048, device=DEVICE)
        self.device = DEVICE
        self.retina.eval()
        self.facenet = InceptionResnetV1(pretrained=facenet_model).eval()
        # Preallocate class attributes
        self.confidence = confidence_threshold
        self.image = None
        self.aligned = None
        self.boxes = None
        self.probs = None
        self.landmarks = None
    
    def detect(self, image_path, cropped_dir=None):
        # Load image using the PIL library and extract useful
        # meta data on the image to save
        self.image, self.image_name, self.image_ext = utils.load_image(image_path)
        self.pixels = np.array(self.image)
        # Extract bounding boxes
        predictions = self.retina.predict_jsons(self.pixels, confidence_threshold=self.confidence)
        # Unpack predictions
        self.boxes = [prediction['bbox'] for prediction in predictions]
        self.probs = [prediction['score'] for prediction in predictions]
        self.landmarks = [prediction['landmarks'] for prediction in predictions]
        # If there are no faces detect, return empty lists
        if len(self.boxes[0]) == 0:
            self.boxes = []
            self.probs = []
            self.landmarks = []
        
        # Save a cropped image of the face?
        if cropped_dir is not None:
            for i,box in enumerate(self.boxes):
                cropped_image = self.image.crop(box)
                cropped_image.save(f'{cropped_dir}/{self.image_name}_{i}{self.image_ext}')
        
        return self.boxes,self.probs,self.landmarks
    
    def _post_process(self, bboxes):
        self.aligned = []
        for bbox in bboxes:
            # Extract aligned face tensor and standardize
            image_tensor = fixed_image_standardization(extract_face(self.image, np.array(bbox)))
            self.aligned.append(image_tensor)
    
    def encode(self):
        '''
        Encodes detected faces using the FaceNet model.

        Returns:
            list: List of encodings or None if no faces are detected.
        '''
        if self.image == None:
            raise Exception('No image detected. Please run RetinaDetector.detect() and try again.')
        
        # Encode. First, extract standardized face tensors
        self._post_process(self.boxes)

        if self.aligned is not None:
            # Prepare (stack) multiple faces for encoding.
            faces = torch.stack([face for face in self.aligned])
            return self.facenet(faces).detach().tolist()
    
    def detect_encode(self, image_path, cropped_dir=None):
        '''
        Detects and encodes an image on disk, and returns a dictionary
        of the results.
        
        Args:
            image_path (str): Path to an image file.
            cropped_dir (str, optional): Directory for saving cropped faces. 
                Default is None.
        
        Returns:
            dict: Dictionary with the detection/encoding results for
                  the image.
        '''
        boxes,probs,landmarks = self.detect(image_path, cropped_dir)
        faces = []
        if len(boxes) == 0:
            faces = None
        else:
            encodings = self.encode()
            if len(boxes) == len(probs) == len(encodings):
                for i in range(len(boxes)):
                    faces.append({
                            'face_id': f'{self.image_name}_{i}',
                            'image_name': self.image_name,
                            'image_path': image_path,
                            'bbox': boxes[i],
                            'prob': probs[i],
                            'landmarks': landmarks[i],
                            'encodings': encodings[i]
                        })
        return faces
