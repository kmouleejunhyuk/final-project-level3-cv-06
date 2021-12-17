# import sys
# sys.path.append('/opt/ml/finalproject')

from torch import torch, nn
import numpy as np
import pickle
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.timer import timer


#dependency
from multilabel.API.preprocess import processer
from multilabel.API.model import val_transform
from multilabel.baseline.model import multihead_hooked
from app.app_config import config as CONFIG
from multilabel.baseline.metrics import top_k_labels

MODELS = CONFIG.multilabel_model
DEVICE = CONFIG.device
LABELS = CONFIG.classes

def cosine_similarity(xray_feat: np.ndarray, image_feat: np.ndarray) -> float:
    '''
    두 feature map간 cosine 유사도를 구해주는 코드
    '''
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


def get_feature(feature_path: str = MODELS.root + MODELS.grad_cam_density_path)->np.ndarray:
    '''
    training set의 모델 feature map 평균을 불러오는 코드
    Args:
        density_path: str
            file 경로
    return:
        xray_mean_feat: np.ndarray
            모델의 feature map 평균 
    '''
    with open(feature_path, 'rb') as f:
        xray_mean_feat = pickle.load(f)

    # print('density loaded')
    return xray_mean_feat.cpu().numpy()


def get_OOD_gradcam_model(weight_path: str = MODELS.root + MODELS.save_path, device: str = DEVICE):
    model = multihead_hooked(38, 6, device)
    model = model.to(device)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.eval()
    # print('model loaded')

    return model


def get_image_from_activation(image, act, grads):
    '''
    activation을 시각화해서 image와 같이 출력하는 코드
    Args:
        image: torch.Tensor
            모델 인퍼런스시 사용한 image, transform 등등 전부 적용되어 있어야 함

        act: torch.Tensor
            모델 특정 레이어의 feature map
        
        grads: torch.Tensor
            모델 특정 레이어의 gradient 분포

    return:
        fig: plt.figure
            이미지에 gradcam을 더해 시각화한 figure
    '''
    pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
    for i in range(act.shape[1]):
        act[:,i,:,:] += pooled_grads[i]

    heatmap_j = torch.mean(act, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max

    
    heatmap_j = resize(heatmap_j, (512,512), preserve_range=True)
    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap_cmap = cmap(heatmap_j, alpha = 0.5)
    
    fig, axs = plt.subplots(1,1,figsize = (5,5), tight_layout = True)
    im = (image*0.2+0.5).permute(1,2,0).cpu()
    axs.imshow(im.cpu())
    axs.imshow(heatmap_cmap)
    axs.axis('off')
    
    return figure_to_array(fig)


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

@timer
def OOD_inference(model: nn.Module, xray_density: np.ndarray, image, device: str= DEVICE):
    '''
    OOD + gradcam 기능이 합쳐진 모델 인퍼런스 코드

    Args:
        model: nn.Module
            multihead_hooked 모델

        xray_density: np.ndarray
            get_density의 output

    returns:
        pred: list(str)
            예측한 레이블 이름

        similarity: float
            사진이 xray와 유사한 정도(float), THR 정의 필요(약 0.5)

        grad_fig: plt.figure
            gradcam figure(no axis)
    '''
    image = image.convert('RGB')
    image = np.array(image)
    process = processer()
    image = process.preprocess(image)
    image = np.array(image)
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
    grad_arr = get_image_from_activation(transformed_image.squeeze(0), grad_activation, grads)

    # label treatment
    category_names = CONFIG.classes
    pred = torch.argmax(out, dim = -1).cpu().detach()
    pred_idx = np.where(pred==1)[1]
    pred = [category_names[cat_id] for cat_id in pred_idx]

    return pred, similarity, grad_arr


#testcode
if __name__ == '__main__':
    import PIL.Image as Image
    image = Image.open('/opt/ml/tmp/img/spanner.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = get_OOD_gradcam_model('/opt/ml/finalproject/multilabel/baseline/save/multi-head_celoss_fulltrain_best.pth', 'cuda')
    xray_mean_feat = get_feature('/opt/ml/tmp/featuremap.pickle')
    pred, similarity, grad_arr = OOD_inference(model, xray_mean_feat, image, 'cuda')
    print(pred, similarity)
    print(type(grad_arr))
    
    Image.fromarray(grad_arr).save('/opt/ml/tmp/f2.png')
