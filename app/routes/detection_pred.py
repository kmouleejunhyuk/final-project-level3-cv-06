import io
import os
from typing import List

from app.app_config import config as CONFIG
from app.customrouter import fileRouter
from fastapi import File, UploadFile
from PIL import Image

from detection.API.model import get_detection_prediction, model_loader

IMG_PATH = CONFIG.static.directiory + "/" + CONFIG.multilabel_model.image_path
LABELS = CONFIG.classes

try:
    MODEL = model_loader()
except:
    raise Exception("detection model loader Error")

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

@router.post("/")
async def get_detection(files: List[UploadFile] = File(...)):
    offset = len(os.listdir(IMG_PATH))

    predictions = []
    for idx, file in enumerate(files):
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))
        pred = get_detection_prediction(MODEL, image, score_threshold=0.8)
        predictions.append(pred)

        filename, ext = os.path.splitext(file.filename)
        file_id = f"{offset+idx:06d}"
        # filename = file_id + 

    return predictions[0].tolist()
