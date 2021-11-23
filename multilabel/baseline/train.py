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

import wandb
from dataset import (
    CustomDataLoader,
    train_transform,
    val_transform
)
from losses import create_criterion
from optim_sche import get_opt_sche
# from utils.utils import add_hist, grid_image, label_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score



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

def calculate_metrics(pred, target):
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def createDirectory(save_dir):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Failed to create the directory.")


def train(model_dir, config_train, thr=0.5):
    seed_everything(config_train['seed'])

    save_dir = increment_path(os.path.join(model_dir, config_train['name']))
    createDirectory(save_dir)

    # settings
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.device_count())

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    from dataset import CustomDataLoader, train_transform, val_transform

    # dataset
    train_dataset = CustomDataLoader(
        image_dir=config_train['image_path'], data_dir=config_train['val_path'], mode="train", transform=train_transform
    )
    val_dataset = CustomDataLoader(
        image_dir=config_train['image_path'], data_dir=config_train['val_path'], mode="val", transform=val_transform
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

    # with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    #     json.dump(vars(config_train), f, ensure_ascii=False, indent=4)

    # start train
    best_val_acc = 0 
    best_val_loss = np.inf
    step = 0
    for epoch in range(config_train['epochs']):
        # train loop
        cal = 0
        model.train()
        loss_value = 0
        matches = 0
        acc = 0
        for idx, train_batch in enumerate(tqdm(train_loader)):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            # print(outs.shape, type(outs))
            pred = np.array(outs.detach().cpu().numpy() > 0.5, dtype = float)
            loss = criterion(outs, labels.type(torch.float))

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            # print(pred, labels)
            # print((pred == labels).sum())
            matches += (pred == labels.detach().cpu().numpy()).sum().item()
            if (idx + 1) % config_train['log_interval'] == 0:
                cal+=1
                train_loss = loss_value / config_train['log_interval']
                train_acc = matches / config_train['batch_size'] / config_train['log_interval'] / n_classes
                result = calculate_metrics(pred, labels.detach().cpu().numpy())
                current_lr = get_lr(optimizer)
                acc += train_acc / 100
                print(
                    f"Epoch[{epoch}/{config_train['epochs']}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {acc*100/cal:4.2%} || lr {current_lr}"
                )
                print(
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(
                                              result['micro/f1'],
                                              result['macro/f1'],
                                              result['samples/f1']))

                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                # wandb log
                if config_train['wandb'] == True:
                    wandb.log(
                        {
                            # "Media/train predict images": figure,
                            "Train/Train loss": round(train_loss, 4),
                            "Train/Train acc": round(result['micro/f1'], 4),
                            "learning_rate": current_lr,
                            "epoch" : epoch+1
                        },
                        step=step,
                    )
                loss_value = 0
                matches = 0
            step += 1

        print(
                f"Epoch[{epoch}/{config_train['epochs']}]({idx + 1}/{len(train_loader)}) || "
                f"training loss {train_loss:4.4} || training accuracy {acc*100/cal:4.2%} || lr {current_lr}"
            )

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss = 0
            val_acc = 0
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(device)

                outs = model(inputs)
                pred = np.array(outs.detach().cpu().numpy() > 0.5, dtype = float)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels.detach().cpu().numpy() == pred).sum().item()
                val_loss += loss_item
                val_acc += acc_item

            val_loss = val_loss / len(val_loader)
            val_acc = val_acc / len(val_dataset) / n_classes
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            # wandb log
            if config_train['wandb'] == True:
                wandb.log(
                    {
                        # "Media/train predict images": figure,
                        "Valid/Valid loss": round(val_loss, 4),
                        "Valid/Valid acc": round(val_acc, 4),
                        "epoch": epoch+1
                    },
                    step=step,
                )
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_train', default="config/train.yaml",type=str, help='path of train configuration yaml file')

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

    train(model_dir, config_train)
