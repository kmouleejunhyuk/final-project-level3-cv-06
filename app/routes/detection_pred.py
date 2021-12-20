import io
import os
from typing import List

import numpy as np
from app.app_config import config as CONFIG
from app.customrouter import fileRouter
from fastapi import File, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

from detection.API.model import get_detection_prediction_toLabel, model_loader

IMG_PATH = CONFIG.static.directiory + "/" + CONFIG.detection_model.image_path
LABELS = CONFIG.classes

try:
    MODEL = model_loader()
except:
    raise Exception("detection model loader Error")

router = fileRouter(prefix=__file__)

ITEMS = {}
for file in os.listdir(IMG_PATH):
    if file[0] == ".": continue
    filename, ext = os.path.splitext(file)
    name, pred = filename[:6], filename[6:]
    ITEMS[name] = pred

@router.get("/")
def get_item():
    # 전체 예측 정보
    return ITEMS

@router.get("/{img_id}")
def get_item_by_img_id(img_id):
    # img_id 기준으로 예측정보 출력
    return ITEMS[img_id]

@router.get("/image/{img_id}", response_class=RedirectResponse)
def get_image_by_img_id(img_id):
    pred = get_item_by_img_id(img_id)
    return "/" + IMG_PATH + "/" + img_id + pred + ".png"

@router.post("/")
async def get_detection(files: List[UploadFile] = File(...)):
    offset = len(os.listdir(IMG_PATH))

    for idx, file in enumerate(files):
        image_bytes = await file.read()
        image, pred = get_detection_prediction_toLabel(MODEL, image_bytes, score_threshold=0.8)

        filename, ext = os.path.splitext(file.filename)
        file_id = f"{offset+idx:06d}"
        filename = file_id + str(pred) + ext

        img_arr = np.array(image)
        detect_image = Image.fromarray(img_arr.astype('uint8'))
        detect_image.save(os.path.join(IMG_PATH, filename))

        ITEMS[file_id] = str(pred)

    return file_id

