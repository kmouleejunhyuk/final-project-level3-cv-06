from front.utils.utils import *
from front.utils import distance as dst
from front.models import VGGFace

import os
import numpy as np
import cv2
import moviepy.editor as mp
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

def process_video(video_path, known_face, threshold=0.35, user_name="dohyun"):
    default_path = os.getcwd()
    output = '/video_' + video_path.split('/')[-1]
    audio = '/audio_' + video_path.split('/')[-1].split('.')[0] + '.wav'

    # extract audio
    video = mp.VideoFileClip(default_path + video_path)
    video.audio.write_audiofile(default_path + audio)

    # write video info
    cap = cv2.VideoCapture(default_path + video_path)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(default_path+output, fourcc, fps, (w, h))
    
    frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    while True:
        ret, frame = cap.read()
        count += 1
        try:
            face_images, face_locations = face_detect(frame)
            face_images = face_preprocess(face_images)

            for i in range(len(face_images)):
                face_encoding = model.predict(face_images[i])[0].tolist()

                # calculate distance (Cosine Distance / Euclidean Distance)
                distance = dst.findCosineDistance(known_face, face_encoding)
                distance = np.float64(distance) # causes trouble for euclideans in api calls if this is not set (issue #175)

                # decision using threshold
                identified = True if distance <= threshold else False
                name = user_name if identified==True else 'Unknwon'

                # mosaic
                x, y, w, h = face_locations[i]
                top, right, bottom, left = y+h, x+w, y, x

                if name == 'Unknwon':
                    frame[bottom:top, left:right, :] = mosaic(frame[bottom:top, left:right, :])

                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)
        
        except Exception as e:
            print(e)
            pass
        
        # show the frame
        # cv2.imshow('Check', frame)
        out.write(frame)
        if count == frame_length:
            break

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # clear program and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('All done!')

    # merge video and audio
    fin_video = default_path + '/output.mp4'
    detected_video = mp.VideoFileClip(default_path + output)
    detected_video.audio = mp.AudioFileClip(default_path + audio)
    detected_video.write_videofile(fin_video)

    return fin_video
