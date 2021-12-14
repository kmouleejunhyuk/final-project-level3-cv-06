# import sys
# sys.path.append('/opt/ml/finalproject')

from torch import torch, nn
import numpy as np
import pickle
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt

#dependency
from multilabel.API.preprocess import processer
from multilabel.API.model import val_transform
from multilabel.baseline.model import multihead_hooked
from app.app_config import config as CONFIG
from multilabel.baseline.metrics import top_k_labels


def cosine_similarity(xray_feat: np.ndarray, image_feat: np.ndarray) -> float:
    assert xray_feat.shape == image_feat.shape
    xray_feat_squared = xray_feat**2
    image_feat_squared = image_feat**2

    xray_l2 = np.sqrt(
        np.sum(xray_feat_squared.ravel())
    )
    image_l2 = np.sqrt(
        np.sum(image_feat_squared.ravel())
    )

    rad = np.sum((xray_feat * image_feat).ravel())

    return np.abs(rad / (xray_l2 * image_l2))


def get_density(density_path: str)->np.ndarray:
    with open(density_path, 'rb') as f:
        xray_mean_feat = pickle.load(f)

    print('density loaded')
    return xray_mean_feat.cpu().numpy()


def get_OOD_gradcam_model(weight_path: str, device: str):
    model = multihead_hooked(38, 6, device)
    model = model.to(device)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.eval()
    print('model loaded')

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


def OOD_inference(model: nn.Module, xray_density: np.ndarray, image: np.ndarray, device: str):
    '''
    model: multihead_hooked 모델
    pred: 예측한 레이블 이름(list(str))
    similarity: 사진이 xray와 유사한 정도(float), THR 정의 필요(약 800)
    grad_fig: gradcam figure(no axis)
    '''
    process = processer()
    image = process.preprocess(image)
    transformed_image = val_transform(image = image)['image'].unsqueeze(0).to(device)
    criterion = nn.CrossEntropyLoss()

    out, cls_out = model(transformed_image)

    #OOD detection(THR: 0.5)
    density_activation = model.OODhook.pop()
    similarity = cosine_similarity(xray_density, density_activation.squeeze(0).cpu().numpy())

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
    grad_fig = get_image_from_activation(transformed_image.squeeze(0), grad_activation, grads)

    # label treatment
    category_names = CONFIG.classes
    pred = torch.argmax(out, dim = -1).cpu().detach()
    pred_idx = np.where(pred==1)[1]
    pred = [category_names[cat_id] for cat_id in pred_idx]

    return pred, similarity, grad_fig


#testcode
if __name__ == '__main__':
    import cv2
    image = cv2.imread('/opt/ml/tmp/img/spanner.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = get_OOD_gradcam_model('/opt/ml/finalproject/multilabel/baseline/save/multi-head_celoss_fulltrain_best.pth', 'cuda')
    xray_mean_feat = get_density('/opt/ml/tmp/featuremap.pickle')
    pred, similarity, grad_fig = OOD_inference(model, xray_mean_feat, image, 'cuda')
    print(pred, similarity)
    grad_fig.savefig('/opt/ml/tmp/fig2.png', dpi=300, facecolor='#eeeeee', bbox_inches='tight', pad_inches = 0)
