from front.utils.utils import *
from front.utils import distance as dst
from front.models import VGGFace

import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings(action='ignore')

model = VGGFace.load_Model
model = model()

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def process_image(img_path):
    known_image = load_image(img_path)
    known_face_image, known_face_location = face_detect(known_image)
    known_face_image = face_preprocess(known_face_image)
    knwon_face_encoding = model.predict(known_face_image)[0].tolist()
    return knwon_face_encoding
