import argparse
import yaml
import glob
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

import wandb
from dataset import CustomDataLoader
from dataset import train_transform
from losses import create_criterion
from optim_sche import get_opt_sche
from metrics import get_metrics_from_matrix, top_k_labels, get_confusion_matrix
from visualize import draw_batch_images
import shutil
import time


def print_size_of_model(model):
    model_path = "/opt/ml/tmp/model.p"
    torch.save(model.state_dict(), model_path)
    print('size(mb) : ', os.path.getsize(model_path) / 1e6)
    os.remove(model_path)


@torch.no_grad()
def test_model(model, testloader, device, half=False):
    total = 0
    model = model.to(device)
    start_time = time.time()
    total_emr = []
    for data in tqdm(testloader, desc = f'quant: {half}'):
        images, labels = data
        images = images.to(device)
        if half:
          images = images.half()
        outputs, _ = model(images)
        predicted = torch.argmax(outputs, dim = -1).cpu().detach()
        total += labels.size(0)
        total_emr.append(np.mean((predicted == labels).min(axis = -1).values.numpy()))
    print(f'EMR of the network on test images: {100*np.mean(total_emr)}%')
    print(f'Elpased time: {round(time.time() - start_time, 3)}s, on {device}')
    

def quant_eval(model_dir, config_train):
    fp32_path = os.path.join(model_dir, config_train['name']) + '/last.pth'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_dataset = CustomDataLoader(
        image_dir=config_train['image_path'], 
        data_dir=config_train['train_path'],
        mode="sampled", 
        transform=train_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_train['batch_size'],
        num_workers=config_train['workers'],
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    N_CLASSES = 38
    model_module = getattr(
        import_module("model"), 
        config_train['model']
    )
    model = model_module(
        num_classes=N_CLASSES, 
        cls_classes = 6, 
        device = device
    )
    model = model.to(device)
    state_dict = torch.load(fp32_path)
    model.load_state_dict(state_dict)

    model.eval()
    print("[fp32]")
    print_size_of_model(model)
    test_model(
        model=model,
        testloader=train_loader,
        device=device,
        half=False
    )
    

    model_module = getattr(
        import_module("model"), 
        config_train['model']
    )
    model = model_module(
        num_classes=N_CLASSES, 
        cls_classes = 6, 
        device = device
    )
    model = model.to(device)
    state_dict = torch.load(fp32_path)
    model.load_state_dict(state_dict)
    model.eval()
    fp16_model = model.half()
    print("[fp16]")
    print_size_of_model(fp16_model)
    test_model(
        model=fp16_model,
        testloader=train_loader,
        device=device,
        half=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_train',
        default = '/opt/ml/runs/mseloss_clsupgrade_multihead_with_quant/multihead_train.yaml', 
        type=str, 
        help='path of train configuration yaml file'
    )

    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    quant_eval(config_train['model_dir'], config_train)