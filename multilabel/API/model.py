import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from app.app_config import config as CONFIG
from imgaug.augmenters.size import pad
from multilabel.baseline import model as mlmodels
from utils.timer import timer

MODELS = CONFIG.multilabel_model
DEVICE = CONFIG.device
LABELS = CONFIG.classes


def model_loader():
    model_class = getattr(mlmodels, MODELS.name)
    model = model_class(MODELS.num_classes, MODELS.cls_classes, DEVICE)

    state_dict = torch.load(MODELS.root + MODELS.save_path)
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    return model


class ratio_aware_pad(ImageOnlyTransform):
    def __init__(self, padmax = None):
        #usage: 
        #  A.Compose([ratio_aware_pad(padmax = 2000)])
        #  maximum size pad(output: 2000 * 2000 * 3), may be sparse

        #usage2: 
        #   A.Compose([ratio_aware_pad(), A.Resize(512, 512)])
        #   ratio aware pad, less sparse, must be resized(size may differ by image)

        super().__init__(always_apply=True)
        self.padmax = padmax

    def apply(self, img, **params):
        if self.padmax:
            #max size aware padding
            #크기 이상인 이미지는 없다고 가정
            assert img.shape[0] < self.padmax and img.shape[1] < self.padmax
            u, r = (self.padmax - img.shape[0]) // 2, (self.padmax - img.shape[1]) // 2
            d, l = self.padmax - img.shape[0] - u, self.padmax - img.shape[1] - r
            img = pad(img, top = u, bottom = d, right = r, left = l, cval = 250)
            
        else:   #ratio-aware padding
            h, w = img.shape[0], img.shape[1]
            if h == w: return img

            if h > w:
                r_delta = (h - w) // 2
                l_delta = (h - w) - r_delta
                img = pad(img, right = r_delta, left = l_delta, cval = 250)
            else:
                u_delta = (w - h) // 2
                d_delta = (w - h) - u_delta
                img = pad(img, top = u_delta, bottom = d_delta, cval = 250)

            assert img.shape[0] == img.shape[1]
        return img


val_transform = A.Compose([
    ratio_aware_pad(),
    A.Resize(512, 512),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])


@timer
def get_multilabel_prediction(model, image):
    '''
    return tensor ex) [0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0]
    '''
    image = image.convert('RGB')
    image = np.array(image)
    image = val_transform(image = image)['image'].unsqueeze(0).to(DEVICE)
    out, _ = model(image)
    pred = torch.argmax(out, dim = -1).cpu().detach()

    return pred[0]


def get_multilabel_prediction_toLabel(model, image):
    '''
    return label ex) [knife, usb, ...]
    '''
    pred = get_multilabel_prediction(model, image)

    labels = []
    for label, bool in zip(LABELS, pred):
        if bool:
            labels.append(label)

    return labels


def get_multilabel_prediction_toindex(model, image):
    '''
    return index ex) [10, 15, 30, ...]
    '''
    pred = get_multilabel_prediction(model, image)

    index = []
    for idx, bool in enumerate(pred):
        if bool:
            index.append(idx)

    return index


def get_multilabel_prediction_toindex_toLabel(model, image):
    '''
    return index and label
    '''
    pred = get_multilabel_prediction(model, image)

    index, labels = [], []
    for idx, (label, bool) in enumerate(zip(LABELS, pred)):
        if bool:
            index.append(idx)
            labels.append(label)
    
    return index, labels
