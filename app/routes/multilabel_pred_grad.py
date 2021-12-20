from fastapi import File, UploadFile
from fastapi.responses import RedirectResponse
from typing import List

import os
import io
from PIL import Image

from app.customrouter import fileRouter
from app.app_config import config as CONFIG
from multilabel.API.model import model_loader, get_multilabel_prediction_toindex
from multilabel.API.OOD import get_OOD_gradcam_model, get_feature, OOD_inference

# IMG_PATH = os.path.join(CONFIG.static.directiory, CONFIG.multilabel_model.image_path)
IMG_PATH = CONFIG.static.directiory + "/" + CONFIG.multilabel_model.image_path
# print(IMG_PATH)


try:
    MODEL = get_OOD_gradcam_model()
    GRAD_CAM_DENSITY = get_feature()
except:
    raise Exception("multilabel model loader Error")

router = fileRouter(prefix=__file__)

ITEMS = {}
for file in os.listdir(IMG_PATH):
    filename, ext = os.path.splitext(file)
    name, pred = filename[:6], filename[6:]
    ITEMS[name] = pred


@router.post("/")
async def upload_files(file: UploadFile = File(...)):
    offset = len(os.listdir(IMG_PATH))

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    # pred = get_multilabel_prediction_toindex(MODEL, img)
    pred, similarity, grad_fig = OOD_inference(MODEL, GRAD_CAM_DENSITY, img)
    print(pred, similarity)

    filename, ext = os.path.splitext(file.filename)
    file_id = f"{offset:06d}"
    filename = file_id + str(pred) + ext
    with open(os.path.join(IMG_PATH, filename), "wb") as fp:
        fp.write(contents)
    ITEMS[file_id] = str(pred)

    return file_id