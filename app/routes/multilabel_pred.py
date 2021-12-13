from fastapi import File, UploadFile
from typing import List

import os
import io
from PIL import Image

from app.customrouter import fileRouter
from app.app_config import config as CONFIG
from multilabel.API.model import model_loader, get_multilabel_prediction_toindex

IMG_PATH = os.path.join(CONFIG.static.directiory, CONFIG.multilabel_model.image_path)

try:
    MODEL = model_loader()
except:
    raise Exception("multilabel model loader Error")

router = fileRouter(prefix=__file__)

ITEMS = {}
for file in os.listdir(IMG_PATH):
    filename, ext = os.path.splitext(file)
    name, pred = int(filename[:6]), filename[6:]
    ITEMS[name] = pred

@router.get("/")
def root():
    # 전체 예측 정보
    return ITEMS

@router.get("/{img_id}")
def root(img_id):
    # 이미지 기준으로 예측정보 출력
    return ITEMS[img_id]

@router.post("/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    offset = len(os.listdir(IMG_PATH))

    predictions = []
    for idx, file in enumerate(files):
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        pred = get_multilabel_prediction_toindex(MODEL, img)

        filename, ext = os.path.splitext(file.filename)
        filename = f"{offset+idx:06d}" + str(pred) + ext
        with open(os.path.join(IMG_PATH, filename), "wb") as fp:
            fp.write(contents)
        predictions.append({file.filename : pred})
        ITEMS[str(offset+idx)] = str(pred)

    return {"prediction": predictions}
