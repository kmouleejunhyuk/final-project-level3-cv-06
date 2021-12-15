
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from app.app_config import config as CONFIG
from detection.models.fasterrcnn import DetectionModel
from utils.timer import timer

MODELS = CONFIG.detection_model
DEVICE = CONFIG.device
LABELS = CONFIG.classes


def model_loader():
    chk_path = MODELS.root + MODELS.save_path

    model = DetectionModel().load_from_checkpoint(chk_path).to(DEVICE)
    model.eval()

    return model


def read_img(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    return img


test_transform = A.Compose([
        ToTensorV2()
    ])


@timer
def get_detection_prediction(model, image, score_threshold: float=0.7):
    '''
    return image
    '''
    image = read_img(image)
    image = test_transform(image=image)['image']
    image = image.unsqueeze(dim=0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        labels = output[0]['labels'].detach().cpu().numpy()
        scores = output[0]['scores'].detach().cpu().numpy()
        boxes = output[0]['boxes'].detach().cpu().numpy()
        indexes = np.where(scores >= score_threshold)

        labels = labels[indexes]
        scores = scores[indexes]
        boxes = boxes[indexes]
    
    image = image.squeeze(dim=0).cpu()
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(len(labels)):
        box = list(map(int, boxes[i]))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (250, 0, 50), 4)

        text = LABELS[labels[i]-1]
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)
        cv2.rectangle(image, (box[0], box[1]-h), (box[0]+w, box[1]), (250, 0, 50), -1)
        cv2.putText(image, text, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
    
    return image
