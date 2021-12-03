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
from losses import create_criterion
from optim_sche import get_opt_sche
from metrics import get_metrics_from_matrix, top_k_labels, get_confusion_matrix
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


def train(model_dir, config_train, config_dir):
    # settings
    seed_everything(config_train['seed'])
    save_dir = increment_path(os.path.join(model_dir, config_train['name']))
    createDirectory(save_dir)
    shutil.copyfile(config_dir, os.path.join(save_dir, config_dir.split('/')[-1]))
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))
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
    N_CLASSES = 38
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
        train_emr = []
        train_confusion_matrix = np.zeros((38, 4))
        for (images, labels) in tqdm(train_loader, desc = f'train/epoch {epoch}', leave = False, total=len(train_loader)):
            images = images.to(device)
            
            optimizer.zero_grad()
            outs, cls_outs = model(images)

            loss = model.get_loss(
                outs, 
                cls_outs, 
                labels, 
                criterion
            )

            preds = top_k_labels(outs, cls_outs)
            
            loss.backward()
            optimizer.step()
            
            # EMR/loss
            images = images.detach().cpu()
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            matrix = get_confusion_matrix(preds, labels)
            train_confusion_matrix += np.array(matrix)
            train_emr.append(np.mean((preds == labels).min(axis = 1)))

            if config_train['wandb'] == True:
                wandb_log = {}
                wandb_log["Train/EMR"] = np.mean(train_emr)
                wandb_log["Train/loss"] = round(loss.item(), 4)
                wandb.log(wandb_log, step)
            step += 1
        
        if scheduler:
            scheduler.step()

        # mAR, mAP, mF1, etc
        if config_train['wandb'] == True:
            wandb_log = {}
            _, metrics = get_metrics_from_matrix(train_confusion_matrix)
            wandb_log["Train/mAR"] = metrics[0]
            wandb_log["Train/mAP"] = metrics[1]
            wandb_log["Train/mF1"] = metrics[2]

            wandb_log["Train/epoch"] = epoch + 1
            wandb_log["learning_rate"] = get_lr(optimizer)
            wandb_log["Image/train image"] = draw_batch_images(images, labels, preds, category_names)
            wandb.log(wandb_log, step)

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_epoch_loss = 0
            val_confusion_matrix = np.zeros((38, 4))
            val_len = len(val_loader)
            valid_emr = []

            for (images, labels) in tqdm(val_loader, desc = f'val/epoch {epoch}', leave = False, total=val_len):
                images = images.to(device)
                
                outs, cls_outs = model(images)

                loss = model.get_loss(outs, cls_outs, labels, criterion)
                preds = top_k_labels(outs, cls_outs)
                
                val_epoch_loss += loss.detach().item()

                # recall, precision, f1
                images, preds, labels = images.detach().cpu(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
                matrix = get_confusion_matrix(preds, labels)
                val_confusion_matrix += np.array(matrix)
                valid_emr.append(np.mean((preds == labels).min(axis = 1)))

            val_epoch_loss /= val_len

            best_val_loss = min(best_val_loss, val_epoch_loss)
            valid_emr = np.mean(valid_emr)
            if valid_emr > best_val_EMR:
                print(f"New best model for EMR : {valid_emr:4.2%}! saving the best model..")
                before_file = glob.glob(os.path.join(save_dir, 'best.pth'))
                if before_file:
                    os.remove(before_file[0])
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_EMR = valid_emr
            torch.save(model.state_dict(), f"{save_dir}/last.pth")

                
            if config_train['wandb'] == True:
                label_metric, (mAR, mAP, mF1) = get_metrics_from_matrix(val_confusion_matrix)
                # wandb log
                wandb_log = {}
                wandb_log["Valid/Valid loss"] = round(val_epoch_loss, 4)
                wandb_log["Image/Valid image"] = draw_batch_images(images.detach().cpu(), labels, preds, category_names)
                wandb_log["epoch"] = epoch + 1
                wandb_log["Valid/EMR"] = valid_emr

                wandb_log[f"Valid/Valid mAR"] = round(mAR, 4)
                wandb_log[f"Valid/Valid mAP"] = round(mAP, 4)
                wandb_log[f"Valid/Valid mF1"] = round(mF1, 4)

                for i in range(38):
                    wandb_log[f"Metric/AR_{category_names[i]}"] = label_metric[i, 0]
                    wandb_log[f"Metric/AP_{category_names[i]}"] = label_metric[i, 1]
                    wandb_log[f"Metric/AF1_{category_names[i]}"] = label_metric[i, 2]

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
