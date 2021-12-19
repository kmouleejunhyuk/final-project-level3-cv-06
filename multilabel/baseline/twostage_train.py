import argparse
import yaml
import glob
import os
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import wandb

from dataset import CustomDataset
from losses import create_criterion
from optim_sche import get_opt_sche
from metrics import (
    get_metrics_from_matrix, 
    top_k_labels, 
    get_confusion_matrix
)
from transform import (
    train_transform, 
    val_transform, 
    train_aug_transform
)
from multilabel_utils.utils import (
    draw_batch_images, 
    seed_everything, 
    increment_path, 
    createDirectory,
    copy_config,
    is_cuda
)
from resources import category_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(config_train, config_dir):
    # settings
    model_dir = config_train['model_dir']

    seed_everything(config_train['seed'])
    save_dir = increment_path(os.path.join(model_dir, config_train['name']))
    createDirectory(save_dir)
    copy_config(config_dir, save_dir)

    use_cuda, device = is_cuda()
    identity = True if 'two' in config_train['model'] else False

    # dataset
    if config_train['augmentation']:
        tr_transform = train_aug_transform
    else:
        tr_transform = train_transform
        
    train_dataset = CustomDataset(
        image_dir=config_train['image_path'], 
        data_dir=config_train['train_path'],
        mode="train", 
        transform=tr_transform
    )
    val_dataset = CustomDataset(
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
    model = model_module(num_classes=N_CLASSES, cls_classes = 6, device = device)
    model = model.to(device)

    if config_train['wandb'] == True:
        wandb.watch(model)

    # loss & optimizer
    criterion = create_criterion(config_train['criterion'])

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

            preds = top_k_labels(
                outs, 
                cls_outs, 
                identity = identity
            )
            
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

                loss = model.get_loss(
                    outs, 
                    cls_outs, 
                    labels, 
                    criterion
                )
                preds = top_k_labels(
                    outs, 
                    cls_outs,
                    identity = identity
                )
                
                val_epoch_loss += loss.detach().item()

                # recall, precision, f1
                images = images.detach().cpu()
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

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

    parser.add_argument('--config_train', default='/opt/ml/finalproject/multilabel/baseline/config/multi_head_train.yaml', type=str, help='path of train configuration yaml file')

    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    # wandb init
    if config_train['wandb'] == True:
        wandb.init(entity=config_train['entity'], project=config_train['project'], config=config_train)
        wandb.run.name = config_train['name']
        wandb.config.update(args)

    train(config_train, args.config_train)
