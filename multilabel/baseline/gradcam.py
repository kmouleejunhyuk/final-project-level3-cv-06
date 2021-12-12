import glob
import os
import random
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
from torch import nn
from torchvision import models
from dataset import CustomDataLoader
from dataset import val_transform
from losses import create_criterion
from optim_sche import get_opt_sche
from metrics import get_metrics_from_matrix, top_k_labels, get_confusion_matrix
from visualize import draw_batch_images
import cv2
from skimage.transform import resize
import matplotlib as mpl
import matplotlib.pyplot as plt


category_names = [
    'Aerosol',
    'Alcohol',
    'Awl',
    'Axe',
    'Bat',
    'Battery',
    'Bullet',
    'Firecracker',
    'Gun',
    'GunParts',
    'Hammer',
    'HandCuffs',
    'HDD',
    'Knife',
    'Laptop',
    'Lighter',
    'Liquid',
    'Match',
    'MetalPipe',
    'NailClippers',
    'PortableGas',
    'Saw',
    'Scissors',
    'Screwdriver',
    'SmartPhone',
    'SolidFuel',
    'Spanner',
    'SSD',
    'SupplymentaryBattery',
    'TabletPC',
    'Thinner',
    'USB',
    'ZippoOil',
    'Plier',
    'Chisel',
    'Electronic cigarettes',
    'Electronic cigarettes(Liquid)',
    'Throwing Knife'
]


class multihead(nn.Module):

    def __init__(self, num_classes, cls_classes, device):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fcs = nn.ModuleList([nn.Linear(2048, 2) for _ in range(38)])
        self.device = device

        #option for gradcam
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        self.layerhook.append(self.backbone.layer4.register_forward_hook(self.forward_hook()))

    
    def activations_hook(self,grad):
        self.gradients = grad


    def get_act_grads(self):
        return self.gradients


    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook


    def forward(self, inputs):
        feat = self.backbone(inputs)
        vecs = []
        for fc in self.fcs:
            vec = fc(feat)
            vecs.append(vec)
        
        stack = torch.stack(vecs, axis = 0)
        return stack.permute(1,0,2), 0, self.selected_out


    def get_loss(self, outs, cls_outs, labels, criterion):
        label_binary = self.get_binary_label(labels).to(torch.long) # .to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, torch.max(label, dim=1)[1])
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))

    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape)).to('cuda')
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)


    use_cuda = torch.cuda.is_available()


def get_model(model_weight_dir = '/opt/ml/tmp/multi-head_celoss_fulltrain_best.pth'):
    model = multihead(38, 6, 'cuda')
    model = model.to('cuda')
    state_dict = torch.load(model_weight_dir)
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

    return fig


def get_gradcam(image: np.ndarray):
    device = torch.device("cuda")
    identity = False

    model = get_model()
    criterion = create_criterion("cross_entropy")
    image = val_transform(image = image)['image']

    image = image.to(device)
    outs, cls_outs, act = model(image.unsqueeze(0))
    preds = top_k_labels(outs, cls_outs, identity = identity).to(device)

    loss = model.get_loss(
        outs,
        cls_outs,
        preds,
        criterion
    )

    loss.backward()

    act = act.detach().cpu()
    grads = model.get_act_grads().detach().cpu()
    fig = get_image_from_activation(image, act, grads)

    return fig, preds
    


if __name__ == '__main__':
    path = '/opt/ml/finalproject/data/eval/Astrophysics/Aerosol/Multiple_Categories/H_8481.80-1090_01_703.png'
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, preds = get_gradcam(image)
    preds = [category_names[x] for x in np.where(preds.cpu() == 1)[1]]

    #시각화용.
    print(','.join(preds))
    fig.savefig('/opt/ml/tmp/fig1.png', dpi=300, facecolor='#eeeeee')