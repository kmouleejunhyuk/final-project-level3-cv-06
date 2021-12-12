import cv2
import math
import numpy as np
from PIL import Image
from .distance import findEuclideanDistance

from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN


def load_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mtcnn expects RGB but OpenCV read BGR
    return img_rgb


def alignment(img, left_eye, right_eye):
	#this function aligns given face in img based on left and right eye coordinates
	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye
	
	#find rotation direction
	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#find length of triangle edges
	a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
	
	#apply cosine rule
	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
		
		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree
		
        #rotate base image
		if direction == -1:
			angle = 90 - angle
		
		img = Image.fromarray(img)
		img = np.array(img.rotate(direction * angle))

	return img #return img anyway


# def face_detect(img):
#     detections = face_detector.detect_faces(img)
#     # print(detections) # [{'box': [109, 111, 189, 258], 'confidence': 0.9999929666519165, 'keypoints': {'left_eye': (147, 210), 'right_eye': (230, 213), 'nose': (166, 257), 'mouth_left': (143, 309), 'mouth_right': (217, 313)}}]
    
#     if len(detections) > 0:
#         detection = detections[0]
#         x, y, w, h = detection['box']
#         detected_face = img[int(y):int(y+h), int(x):int(x+w)]

#         keypoints = detection['keypoints']
#         left_eye = keypoints['left_eye']
#         right_eye = keypoints['right_eye']

#         detected_face = alignment(detected_face, left_eye, right_eye)

#         return detected_face, [x, y, w, h]
    
#     else: # if not face detected
#         raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def face_detect(img):
	face_detector = MTCNN()
	detections = face_detector.detect_faces(img)
	# print(detections) # [{'box': [109, 111, 189, 258], 'confidence': 0.9999929666519165, 'keypoints': {'left_eye': (147, 210), 'right_eye': (230, 213), 'nose': (166, 257), 'mouth_left': (143, 309), 'mouth_right': (217, 313)}}]

	detected_faces = []
	face_locations = []
	if len(detections) > 0:
		for detection in detections:
			x, y, w, h = detection['box']
			detected_face = img[int(y):int(y+h), int(x):int(x+w)]

			keypoints = detection['keypoints']
			left_eye = keypoints['left_eye']
			right_eye = keypoints['right_eye']

			detected_face = alignment(detected_face, left_eye, right_eye)
			detected_faces.append(detected_face)
			face_locations.append([x, y, w, h])
		
		return detected_faces, face_locations
	
	else: # if not face detected
		raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")


def face_preprocess(imgs, target_size=(224, 224)): # VGG (224, 224) / OpenFace (96, 96) / Facenet (160, 160) / DeepID (47, 55)
	imgs_pixs = []
	for img in imgs:		
		# post-processing
		img = cv2.resize(img, target_size)
		img_pixs = image.img_to_array(img)
		img_pixs = np.expand_dims(img_pixs, axis=0)
		img_pixs /= 255 # normalize [0,1]
		imgs_pixs.append(img_pixs)
	return imgs_pixs


def find_input_shape(model):
	# face recognition models have different size of inputs
	# my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	input_shape = model.layers[0].input_shape
	
	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]
		
	if type(input_shape) == list: # issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)
	
	return input_shape