from flask import Flask, request
from flask.templating import render_template
from .models.processing import process_image, process_video

import shutil
import warnings
warnings.filterwarnings(action='ignore')

app = Flask(__name__)
app.debug = True

### Main page ###
@app.route('/')
def index():
    return render_template('index.html')

### Face detection ###
@app.route('/face_detect_get')
def face_detect_get():
    return render_template('face_detect_get.html')

@app.route('/face_detect_post', methods=['GET', 'POST'])
def face_detect_post():
    if request.method == 'POST':
        face_image = request.files['face_img']
        face_image.save('./front/static/input/'+ str(face_image.filename))
        face_image_path = './front/static/input/' + str(face_image.filename)

        known_face_encoding = process_image(face_image_path)

        video_file = request.files['object_file']
        video_file.save('./front/static/input/' + str(video_file.filename))
        video_file_path = '/front/static/input/' + str(video_file.filename)

        fin_video = process_video(video_path=video_file_path, known_face=known_face_encoding)
        shutil.copy(fin_video, './front/static/output.mp4')

    return render_template('face_detect_post.html' , detected=fin_video)
