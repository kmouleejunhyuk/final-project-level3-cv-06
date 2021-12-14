from torch import torch, nn
from torchvision import models
import numpy as np
import pickle
from multilabel.API.preprocess import processer
from multilabel.API.model import val_transform
from multilabel.baseline.model import multihead_hooked
from app.app_config import config as CONFIG
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt

from baseline.metrics import top_k_labels


def cosine_similarity(xray_feat, image_feat) -> float:
    assert xray_feat.shape == image_feat.shape
    xray_feat = xray_feat**2
    image_feat = image_feat**2

    xray_l2 = np.sqrt(
        np.sum(xray_feat.ravel())
    )
    image_l2 = np.sqrt(
        np.sum(image_feat.ravel())
    )

    rad = np.sum((xray_feat * image_feat).ravel())

    return np.abs(rad / (xray_l2 * image_l2))


def get_density(density_path: str)->torch.Tensor:
    with open(density_path, 'rb') as f:
        xray_mean_feat = pickle.load(f)

    print('density loaded')
    return xray_mean_feat


def get_OOD_gradcam_model(weight_path: str, device: str):
    model = multihead_hooked(38, 6, device)
    model = model.to(device)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.eval()
    print('loaded')

    return model


def get_image_from_activation(image, act, grads):
    pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
    for i in range(act.shape[1]):
        act[:,i,:,:] += pooled_grads[i]

    heatmap_j = torch.mean(act, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max

    
    heatmap_j = resize(heatmap_j,(512,512),preserve_range=True)
    cmap = mpl.cm.get_cmap('jet',256)
    heatmap_cmap = cmap(heatmap_j,alpha = 0.5)
    
    fig, axs = plt.subplots(1,1,figsize = (5,5))
    im = (image*0.2+0.5).permute(1,2,0).cpu()
    axs.imshow(im.cpu())
    axs.imshow(heatmap_cmap)
    axs.axis('off')

    return fig


def OOD_inference(model: nn.Module, density_dir: str, image: np.ndarray, device: str):
    '''
    model: multihead_hooked 모델
    pred: 예측한 레이블 이름(list(str))
    confidence: 레이블이 맞을 확률(list(float))
    similarity: 사진이 xray와 유사한 정도(float), THR 정의 필요(약 800)
    grad_fig: gradcam figure(no axis)
    '''
    process = processer()
    image = process.preprocess(image)
    transformed_image = val_transform(image = image)['image'].unsqueeze(0).to(device)
    criterion = nn.CrossEntropyLoss()

    out, cls_out = model(transformed_image)

    #OOD detection(THR: 800)
    density_activation = model.OODhook.pop()
    xray_mean_feat = get_density(density_dir).cpu().numpy()
    similarity = cosine_similarity(xray_mean_feat, density_activation)

    #gradcam
    grad_activation = model.selected_out
    preds = top_k_labels(out, cls_out, identity = False).to(device)

    loss = model.get_loss(
        out,
        cls_out,
        preds,
        criterion
    )

    loss.backward()
    grad_activation = grad_activation.detach().cpu()
    grads = model.get_act_grads().detach().cpu()
    grad_fig = get_image_from_activation(image, grad_activation, grads)

    # label inference
    category_names = CONFIG.classes
    pred = torch.argmax(out, dim = -1).cpu().detach()
    pred_idx = np.where(pred==1)[1]
    pred = [category_names[cat_id] for cat_id in pred_idx]
    confidence = [out[:, cat_id, -1].item() for cat_id in pred_idx]
    anti_confidence = [out[:, cat_id, 0].item() for cat_id in pred_idx]
    confidence = [conf / (anti + conf) for conf, anti in zip(confidence, anti_confidence)]

    return pred, confidence, similarity, grad_fig
