import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from models.fasterrcnn import LitModel
from detection.config.detection_config import config as CONFIG

pl.seed_everything(CONFIG.seed)
chk_path = '/opt/ml/finalproject/detection/lightning_logs/version_0/checkpoints/epoch=49-step=32749.ckpt'
model = LitModel().load_from_checkpoint(chk_path)
model.eval()


def read_img(img_path): # --> to util
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    return img

def valid_transform(): # --> to transform
    return A.Compose([
        A.PadIfNeeded(min_height=1500, min_width=1500, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(1024, 1024),
        ToTensorV2()
    ])

# --> to in __name__ == '__main__'
image = read_img('/opt/ml/finalproject/detection/data/sampled/Astrophysics/[Astro]Aerosol/Aerosol/Multiple_Categories/H_8481.80-1090_01_239.png')
transform = valid_transform()
image = transform(image=image)['image']
image = image.unsqueeze(dim=0)

with torch.no_grad():
    output = model(image)
    results = []
    labels = output[0]['labels'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()
    boxes = output[0]['boxes'].detach().cpu().numpy()
    indexes = np.where(scores > 0.27)

    labels = labels[indexes]
    scores = scores[indexes]
    boxes = boxes[indexes]

image = image.squeeze(dim=0)
image = image.permute(1, 2, 0).numpy()
image= (image * 255).astype(np.uint8)

for i in range(len(labels)):

    box = list(map(int, boxes[i]))
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (250, 0, 50), 5)

    text = str(labels[i])
    cv2.rectangle(image, (box[0], box[1] - 30), (box[0] + 200, box[1]), (250, 0, 50), -1)
    cv2.putText(image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 5, cv2.LINE_AA)

plt.imshow(image)
