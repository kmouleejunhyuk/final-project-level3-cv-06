from re import L
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import os
from pathlib import Path
import shutil
import glob
import re

def draw_batch_images(images, labels, preds, category_names):
    mean, std = 0.5, 0.2
    num_examples = len(images)
    n_cols = 4
    fig, axes = plt.subplots(
        nrows= int(np.ceil(num_examples / n_cols)), 
        ncols=n_cols, 
        figsize=(30, int(4*num_examples / n_cols)), 
        constrained_layout=True
    )

    # fig.tight_layout()
    for row_num, ax in zip(range(num_examples), axes.ravel()):
        # Original Image
        image =  (images[row_num]*std*255) + (mean*255)

        label = np.where(labels[row_num]==1)[0]
        label = [ category_names[cat_id] for cat_id in label]

        pred = np.where(preds[row_num]==1)[0]
        pred = [ category_names[cat_id] for cat_id in pred]

        ax.imshow(image.permute(1,2,0).numpy().astype(int))
        ax.set_title(f"gt : {label},\n pred : {pred}")
    return fig


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False): #-->util
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

def createDirectory(save_dir): #-->util
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError:
        print("Error: Failed to create the directory.")


def copy_config(config_dir, save_dir):
    shutil.copyfile(config_dir, os.path.join(save_dir, config_dir.split('/')[-1]))


def is_cuda():
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return use_cuda, device