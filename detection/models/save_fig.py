import cv2
import matplotlib.pyplot as plt
import numpy as np


def inference_figure(imgs, preds, target, classes, save_dir=''):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), tight_layout=True)
    for i in range(len(imgs)):
        image = imgs[i].squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        img = (image * 255).astype(np.uint8)
        img2 = (image * 255).astype(np.uint8)

        pred = preds[i]
        for j in range(len(pred['labels'])):
            box = list(map(int, pred['boxes'][j].cpu()))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (250, 0, 50), 4)

            text = classes[pred['labels'][j].cpu().item()]
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)
            cv2.rectangle(img, (box[0], box[1]-h), (box[0]+w, box[1]), (250, 0, 50), -1)
            cv2.putText(img, text, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
        x1, y1 = i//2, (i*2)%4
        axes[x1, y1].imshow(img)

        tar = target[i]
        for k in range(len(tar['labels'])):
            box = list(map(int, tar['boxes'][k].cpu()))
            cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (250, 50, 50), 4)

            text = classes[tar['labels'][k].cpu().item()]
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)
            cv2.rectangle(img2, (box[0], box[1]-h), (box[0]+w, box[1]), (250, 50, 50), -1)
            cv2.putText(img2, text, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)
        x2, y2 = i//2, ((i*2)%4)+1
        axes[x2, y2].imshow(img2)

    return fig
    
