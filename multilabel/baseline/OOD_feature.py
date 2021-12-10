import torch.nn as nn
import numpy as np

from model import multihead_with_quant
from dataset import val_transform, CustomDataLoader
import pickle
import torch
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


activation = []
def get_activation(name):
    def hook(model, input, output):
        activation.append(output.detach())
    return hook

def get_model(path = r'/opt/ml/runs/mseloss_clsupgrade_multihead_with_quant/best.pth'):
    model_weight_dir = path
    model = multihead_with_quant(38, 6, 'cuda')
    model = model.to('cuda')
    state_dict = torch.load(model_weight_dir)
    model.load_state_dict(state_dict)
    model.eval()
    print('loaded')

    return model

def main(save_dir):
    model = get_model()
    model.backbone.conv1.register_forward_hook(get_activation('first'))

    val_dataset = CustomDataLoader(
        image_dir="/opt/ml/finalproject/data/",
        data_dir="/opt/ml/finalproject/data/train/full_train.json",
        mode="train", #set train to full-train
        transform=val_transform
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=20,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    model.eval()
    feat_sum = torch.zeros((20, 64, 256, 256))
    count = 0
    with torch.no_grad():
        for (images, labels) in tqdm(val_loader):
            images = images.to('cuda')

            outs, _ = model(images)

            #extract feat 
            if activation:
                f = activation.pop()
                feat_sum += f.cpu()
                count += 1

        
        #batch*64*256*256
        feat_ = torch.sum(feat_sum, axis = 0) / count

        with open(save_dir, 'wb') as f:
            pickle.dump(feat_, f)


if __name__ == '__main__':
    save_dir = '/opt/ml/tmp/featuremap.pickle'
    main(save_dir)

