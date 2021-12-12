from fastapi import FastAPI
from .models.processing import process_image
import shutil
import warnings
from requests import request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
warnings.filterwarnings(action='ignore')


app = FastAPI()
app.mount("/static", StaticFiles(directory="/opt/ml/finalproject/flask/front/static/"), name="static")
app.debug = True
templates = Jinja2Templates(directory="templates")
root = "/opt/ml/finalproject/flask"

### Main page ###
@app.get("/", response_class=HTMLResponse)
def index(root):
    return templates.TemplateResponse(
        name = 'index.html', 
        context = {"request": request, "id": id}
    )

### Face detection ###
@app.route('/face_detect_get')
def face_detect_get():
    return templates.TemplateResponse('face_detect_get.html')

@app.route('/face_detect_post', methods=['GET', 'POST'])
def face_detect_post():
    if request.method == 'POST':
        face_image = request.files['face_img']
        face_image.save('./front/static/input/'+ str(face_image.filename))
        face_image_path = './front/static/input/' + str(face_image.filename)

        known_face_encoding = process_image(face_image_path)

    return templates.TemplateResponse('face_detect_post.html')
