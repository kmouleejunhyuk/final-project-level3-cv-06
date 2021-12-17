import io
import os
from typing import List

from app.app_config import config as CONFIG
from app.customrouter import fileRouter
from fastapi import File, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
from multilabel.API.preprocess import processer

from multilabel.API.OOD import (OOD_inference, get_feature,
                                get_OOD_gradcam_model)

IMG_PATH = CONFIG.static.directiory + "/" + CONFIG.multilabel_model.image_path
LABELS = CONFIG.classes

try:
    MODEL = get_OOD_gradcam_model()
    GRAD_CAM_DENSITY = get_feature()
except:
    raise Exception("multilabel model loader Error")

router = fileRouter(prefix=__file__)

ITEMS = {"None":None}
for file in os.listdir(IMG_PATH):
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
    img_path = os.path.join(os.getcwd(), IMG_PATH, img_id+ITEMS[img_id]+'.png')
    return img_path, ITEMS[img_id]

@router.get("/image/{img_id}", response_class=RedirectResponse)
def get_image_by_img_id(img_id):
    pred = get_item_by_img_id(img_id)
    return "/" + IMG_PATH + "/" + img_id + pred + ".png"

@router.post("/")
async def get_multilabel(files: List[UploadFile] = File(...)):
    offset = len(os.listdir(IMG_PATH))

    predictions = []
    for idx, file in enumerate(files):
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))

        pred, similarity, grad_arr = OOD_inference(MODEL, GRAD_CAM_DENSITY, image)
        
        predictions.append(pred)
        predictions.append(str(similarity))
        predictions.append(grad_arr.tolist())

        filename, ext = os.path.splitext(file.filename)
        file_id = f"{offset+idx:06d}"
        filename = file_id + str(pred) + ext
        if similarity > 0.5:
            with open(os.path.join(IMG_PATH, filename), "wb") as fp:
                fp.write(file_bytes)
        ITEMS[file_id] = str(pred)
    return predictions
