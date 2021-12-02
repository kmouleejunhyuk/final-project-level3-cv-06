import argparse
import yaml
import glob
import json
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
from losses import create_criterion
from optim_sche import get_opt_sche
from metrics import All_metric, top_k_labels
from visualize import draw_batch_images
import shutil

category_names = ['Aerosol', 'Alcohol', 'Awl', 'Axe', 'Bat', 'Battery', 'Bullet', 'Firecracker', 'Gun', 'GunParts', 'Hammer',
 'HandCuffs', 'HDD', 'Knife', 'Laptop', 'Lighter', 'Liquid', 'Match', 'MetalPipe', 'NailClippers', 'PortableGas', 'Saw', 'Scissors', 'Screwdriver',
 'SmartPhone', 'SolidFuel', 'Spanner', 'SSD', 'SupplymentaryBattery', 'TabletPC', 'Thinner', 'USB', 'ZippoOil', 'Plier', 'Chisel', 'Electronic cigarettes',
 'Electronic cigarettes(Liquid)', 'Throwing Knife']

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def createDirectory(save_dir):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Failed to create the directory.")


def train(model_dir, config_train, config_dir, thr = 0.5):
    seed_everything(config_train['seed'])

    save_dir = increment_path(os.path.join(model_dir, config_train['name']))
    createDirectory(save_dir)
    shutil.copyfile(config_dir, os.path.join(save_dir, config_dir.split('/')[-1]))

    # settings
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.device_count())

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset
    import sys
    sys.path.append('/opt/ml/finalproject/multilabel/baseline')
    from dataset import train_transform, val_transform
    
    train_dataset = CustomDataLoader(
        image_dir=config_train['image_path'], 
        data_dir=config_train['train_path'],
        mode="sampled", 
        transform=train_transform
    )
    val_dataset = CustomDataLoader(
        image_dir=config_train['image_path'], 
        data_dir=config_train['val_path'], 
        mode="eval", 
        transform=val_transform
    )

    # data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_train['batch_size'],
        num_workers=config_train['workers'],
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config_train['batch_size'],
        num_workers=config_train['workers'],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # model
    n_classes = 38
    model_module = getattr(import_module("model"), config_train['model'])
    model = model_module(num_classes=n_classes, cls_classes = 6, device = device)
    model = model.to(device)

    if config_train['wandb'] == True:
        wandb.watch(model)

    # loss & optimizer
    criterion = create_criterion(config_train['criterion'])
    metric_key = ['recall', 'precision', 'f1', 'emr']

    # optimizer & scheduler
    optimizer, scheduler = get_opt_sche(config_train, model)

    # start train
    best_val_EMR = -1
    best_val_loss = np.inf
    step = 0

    for epoch in range(config_train['epochs']):
        # train loop
        model.train()
        train_metric = np.zeros((4, ))
        for idx, (images, labels) in tqdm(enumerate(train_loader), desc = f'train/epoch {epoch}', leave = False, total=len(train_loader)):
            images = images.to(device)
            # labels = labels
            
            optimizer.zero_grad()
            outs, cls_outs = model(images)

            loss = model.get_loss(outs, cls_outs, labels, criterion)
            preds = top_k_labels(outs, cls_outs)
            
            loss.backward()
            optimizer.step()
            
            # acc, recall, precision, auc
            images, preds, labels = images.detach().cpu(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
            metrics, _ = All_metric(preds, labels)
            train_metric += np.array(metrics)
            # wandb log(batch metric)
            if config_train['wandb'] == True:
                wandb_log = {}
                for key, met in zip(metric_key, metrics):
                    wandb_log[f"Train/{key}"] = round(met, 4)
          
                wandb_log["Train/epoch"] = epoch + 1
                wandb_log["Train/loss"] = round(loss.item(), 4)
                wandb_log["learning_rate"] = get_lr(optimizer)
                wandb.log(wandb_log, step)
            step += 1
        
        if scheduler:
            scheduler.step()

        # wandb log(batch metric)
        if config_train['wandb'] == True:
            wandb.log({"Image/train image": draw_batch_images(
                    images, labels, preds, category_names
                )}, step)

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_epoch_loss = 0
            class_val_epoch_metric = np.zeros((38, 3))
            val_metric = np.zeros((4, ))
            for (images, labels) in tqdm(val_loader, desc = f'val/epoch {epoch}', leave = False, total=len(val_loader)):
                images = images.to(device)
                
                optimizer.zero_grad()
                outs, cls_outs = model(images)

                loss = model.get_loss(outs, cls_outs, labels, criterion)
                preds = top_k_labels(outs, cls_outs)
                
                val_epoch_loss += loss.detach().item()

                # recall, precision, f1
                images, preds, labels = images.detach().cpu(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
                _metric, per_label_metric = All_metric(preds, labels)
                class_val_epoch_metric += per_label_metric
                val_metric += np.array(_metric)

            val_epoch_loss /= len(val_loader)
            class_val_epoch_metric /= len(val_loader)
            val_metric /= len(val_loader)

            best_val_loss = min(best_val_loss, val_epoch_loss)
            if val_metric[-1] > best_val_EMR:
                print(f"New best model for EMR : {val_metric[-1]:4.2%}! saving the best model..")
                before_file = glob.glob(os.path.join(save_dir, 'best.pth'))
                if before_file:
                    os.remove(before_file[0])
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_EMR = val_metric[-1]
            torch.save(model.state_dict(), f"{save_dir}/last.pth")

                
            if config_train['wandb'] == True:
                # wandb log
                wandb_log = {}
                wandb_log["Valid/Valid loss"] = round(val_epoch_loss, 4)
                wandb_log["Image/Valid image"] = draw_batch_images(images.detach().cpu(), labels, preds, category_names)
                wandb_log["epoch"] = epoch + 1

                for idx, i in enumerate(metric_key):
                    wandb_log[f"Valid/Valid {i}"] = round(val_metric[idx], 4)

                for i in range(38):
                    wandb_log[f"Metric/Recall_{category_names[i]}"] = class_val_epoch_metric[i, 0]
                    wandb_log[f"Metric/Precision_{category_names[i]}"] = class_val_epoch_metric[i, 1]
                    wandb_log[f"Metric/f1_{category_names[i]}"] = class_val_epoch_metric[i, 2]

                wandb.log(wandb_log, step=step)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_train', default='/opt/ml/finalproject/multilabel/baseline/config/train.yaml', type=str, help='path of train configuration yaml file')

    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    # check_args(args)
    # print(args)

    # wandb init
    if config_train['wandb'] == True:
        wandb.init(entity=config_train['entity'], project=config_train['project'])
        wandb.run.name = config_train['name']
        wandb.config.update(args)

    model_dir = config_train['model_dir']

    train(model_dir, config_train, args.config_train)
