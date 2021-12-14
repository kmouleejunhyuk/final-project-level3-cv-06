from fastapi import File, UploadFile
from fastapi.responses import RedirectResponse
from typing import List

import os
import io
from PIL import Image

from app.customrouter import fileRouter
from app.app_config import config as CONFIG
from multilabel.API.model import model_loader, get_multilabel_prediction_toindex

# IMG_PATH = os.path.join(CONFIG.static.directiory, CONFIG.multilabel_model.image_path)
IMG_PATH = CONFIG.static.directiory + "/" + CONFIG.multilabel_model.image_path
print(IMG_PATH)


try:
    MODEL = model_loader()
except:
    raise Exception("multilabel model loader Error")

router = fileRouter(prefix=__file__)

ITEMS = {}
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
    return ITEMS[img_id]

@router.get("/image/{img_id}", response_class=RedirectResponse)
def get_image_by_img_id(img_id):
    pred = get_item_by_img_id(img_id)
    return "/" + IMG_PATH + "/" + img_id + pred + ".png"

@router.post("/")
async def upload_files(file: UploadFile = File(...)):
    offset = len(os.listdir(IMG_PATH))

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    pred = get_multilabel_prediction_toindex(MODEL, img)

    filename, ext = os.path.splitext(file.filename)
    file_id = f"{offset:06d}"
    filename = file_id + str(pred) + ext
    with open(os.path.join(IMG_PATH, filename), "wb") as fp:
        fp.write(contents)
    ITEMS[file_id] = str(pred)

    return file_id

# 여러 파일 업로드
# @router.post("/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     offset = len(os.listdir(IMG_PATH))

#     result = []
#     for idx, file in enumerate(files):
#         contents = await file.read()
#         img = Image.open(io.BytesIO(contents))
#         pred = get_multilabel_prediction_toindex(MODEL, img)

#         filename, ext = os.path.splitext(file.filename)
#         file_id = f"{offset+idx:06d}"
#         filename = file_id + str(pred) + ext
#         with open(os.path.join(IMG_PATH, filename), "wb") as fp:
#             fp.write(contents)
#         result.append(file_id)
#         ITEMS[file_id] = str(pred)

#     return result
