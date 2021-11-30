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
from metrics import All_metric
from visualize import draw_batch_images
import shutil
import warnings



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
        image_dir=config_train['train_image_path'], 
        data_dir=config_train['train_path'],
        mode="train", 
        transform=train_transform
    )
    val_dataset = CustomDataLoader(
        image_dir=config_train['val_image_path'], 
        data_dir=config_train['val_path'], 
        mode="val", 
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
    model = model_module(num_classes=n_classes)
    model = model.to(device)
    if config_train['wandb'] == True:
        wandb.watch(model)

    # loss & optimizer
    criterion = create_criterion(config_train['criterion'])

    # optimizer & scheduler
    optimizer, scheduler = get_opt_sche(config_train, model)


    # start train
    best_val_acc = 0 
    best_val_loss = np.inf
    step = 0
    for epoch in range(config_train['epochs']):
        # train loop
        model.train()
        epoch_loss = 0
        epoch_metric = np.zeros(5)

        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            outs = model(images)
            preds = torch.where(outs>thr, 1., 0.).detach()
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            
            # acc, recall, precision, f1, auc
            images, preds, labels = images.detach().cpu(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No positive samples in y_true, true positive value should be meaningless")
                iter_metric = All_metric(preds, labels, n_classes, outs.detach().cpu().numpy())
            epoch_metric = epoch_metric + iter_metric

            epoch_loss += loss.item()
            current_lr = get_lr(optimizer)

            if (idx + 1) % config_train['log_interval'] == 0:
                print(
                    f"Epoch[{epoch}/{config_train['epochs']}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {loss:4.4} || training accuracy {iter_metric[0]:4.2%} || lr {current_lr} || "
                    f"training recall {iter_metric[1]:.2} || training precision {iter_metric[2]:.2} || training f1 {iter_metric[3]:.2} || training AUC {iter_metric[4]:.2}"
                    )

            # wandb log
            if config_train['wandb'] == True:
                wandb_log = {}
                for idx, i in enumerate(['acc', 'recall', 'precision', 'f1', 'auc']):
                    wandb_log[f"Train/Train {i}"] = round(iter_metric[idx], 4)                
                wandb_log["Train/epoch"] = epoch + 1
                wandb_log["learning_rate"] = current_lr
                wandb_log["Train/Train loss"] = round(loss.item(), 4)
                wandb.log(wandb_log, step)
            step += 1

            # if (idx+1)==len(train_loader):
        print(f"{epoch} Epoch's overall result")
        print(
            f"Epoch[{epoch}/{config_train['epochs']}]({idx + 1}/{len(train_loader)}) || "
            f"training loss {epoch_loss/len(train_loader):4.4} || training accuracy {epoch_metric[0]/len(train_loader):4.2%} ||"
            f"training recall {epoch_metric[1]/len(train_loader):.2} || training precision {epoch_metric[2]/len(train_loader):.2} || training f1 {epoch_metric[3]/len(train_loader):.2} ||"
            f"training AUC {epoch_metric[4]/len(train_loader):.2}"
            )
        wandb.log({"Image/Train image" : draw_batch_images(images, labels, preds, category_names), "epoch" : epoch+1})
        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_epoch_loss = 0
            class_val_epoch_metric = np.zeros((38, 5))

            for val_batch in val_loader:
                images, labels = val_batch
                images = images.to(device)
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(device)

                outs = model(images)
                preds = torch.where(outs>thr, 1., 0.).detach()

                loss = criterion(outs, labels).item()
                
                val_epoch_loss += loss

                # acc, recall, precision, auc
                preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
                val_iter_metric = All_metric(preds, labels, n_classes, type='val')
                class_val_epoch_metric += val_iter_metric

            val_epoch_loss /= len(val_loader)
            # val_epoch_metric = [i/len(val_loader) for i in val_epoch_metric]

            # val_epoch_metric shape: (5,) 
            val_epoch_metric = np.mean(class_val_epoch_metric, axis=0)
            val_epoch_metric = val_epoch_metric/len(val_loader)

            # class_val_epoch_metric shape: (38, 5)
            class_val_epoch_metric = class_val_epoch_metric/len(val_loader)


            best_val_loss = min(best_val_loss, val_epoch_loss)
            if val_epoch_metric[0] > best_val_acc:
                print(f"New best model for val accuracy : {val_epoch_metric[0]:4.2%}! saving the best model..")
                before_file = glob.glob(os.path.join(save_dir, 'best*.pth'))
                if before_file:
                    os.remove(before_file[0])
                torch.save(model.state_dict(), f"{save_dir}/best_epoch{epoch+1}_{val_epoch_metric[0]:4.2%}.pth")
                best_val_acc = val_epoch_metric[0]
            # torch.save(model.state_dict(), f"{save_dir}/last_epoch{epoch}.pth")
            print(
                f"[Val] acc : {val_epoch_metric[0]:4.2%}, loss: {val_epoch_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            # wandb log
            wandb_log = {}
            wandb_log["Valid/Valid loss"] = round(val_epoch_loss, 4)
            wandb_log["Image/Valid image"] = draw_batch_images(images.detach().cpu(), labels, preds, category_names)
            wandb_log["epoch"] = epoch + 1

            for idx, i in enumerate(['acc', 'recall', 'precision', 'f1', 'auc']):
                wandb_log[f"Valid/Valid {i}"] = round(val_epoch_metric[idx], 4)

            for i in range(38):
                wandb_log[f"Metric_Acc/{category_names[i]}"] = class_val_epoch_metric[i][0]
                wandb_log[f"Metric_Recall/{category_names[i]}"] = class_val_epoch_metric[i][1]
                wandb_log[f"Metric_Precision/{category_names[i]}"] = class_val_epoch_metric[i][2]
                wandb_log[f"Metric_f1/{category_names[i]}"] = class_val_epoch_metric[i][3]
                wandb_log[f"Metric_Auc/{category_names[i]}"] = class_val_epoch_metric[i][4]
                
            if config_train['wandb'] == True:
                wandb.log(wandb_log,
                    step=step,
                )
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
