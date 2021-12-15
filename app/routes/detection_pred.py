import io
import os
from typing import List

from app.customrouter import fileRouter
from fastapi import File, UploadFile
from PIL import Image

from detection.API.model import get_detection_prediction, model_loader

try:
    MODEL = model_loader()
except:
    raise Exception("detection model loader Error")

router = fileRouter(prefix=__file__)

@router.post("/")
async def get_detection(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))
        pred = get_detection_prediction(MODEL, image, score_threshold=0.8)
        predictions.append(pred)
    return predictions[0].tolist()
