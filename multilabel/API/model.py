import numpy as np
import torch
from app.app_config import config as CONFIG
from multilabel.baseline import model as mlmodels
from utils.timer import timer
from multilabel.baseline.transform import val_transform

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


@timer
def get_multilabel_prediction(model, image):
    '''
    return tensor ex) [0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0]
    '''
    image = image.convert('RGB')
    image = np.array(image)
    image = val_transform(image = image)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
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
