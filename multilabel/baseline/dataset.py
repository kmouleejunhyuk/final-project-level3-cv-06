import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import webcolors

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
# from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

import copy

# train_path = '../tmp/modified_tmp_dummy.json' # json path
# dataset_path = '../tmp/sample_images_512' # image path

# category_names = ['Background','Aerosol', 'Alcohol', 'Awl', 'Axe', 'Bat', 'Battery', 'Bullet', 'Firecracker', 'Gun', 'GunParts', 'Hammer',
#  'HandCuffs', 'HDD', 'Knife', 'Laptop', 'Lighter', 'Liquid', 'Match', 'MetalPipe', 'NailClippers', 'PrtableGas', 'Saw', 'Scissors', 'Screwdriver',
#  'SmartPhone', 'SolidFuel', 'Spanner', 'SSD', 'SupplymentaryBattery', 'TabletPC', 'Thinner', 'USB', 'ZippoOil', 'Plier', 'Chisel', 'Electronic cigarettes',
#  'Electronic cigarettes(Liquid)', 'Throwing Knife']
class_num = 38

def make_cls_id(origin_id):
    # 14, 35, 40 label이 없음
    if 0<origin_id<14:
        cat_id = origin_id - 1
        return cat_id

    elif 15<=origin_id<35:
        cat_id = origin_id -3
        return cat_id

    elif 36<=origin_id<41:
        cat_id = origin_id -3
        return cat_id
    elif origin_id==41:
        cat_id = origin_id-4
        return cat_id
    else:
        print('없는 category id 입니다', origin_id)
        return None


class CustomDataLoader(Dataset):
    """
    coco format
    """
    def __init__(self, image_dir, data_dir, mode="train", transform=None, class_num=38):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.class_num = class_num
        self.image_dir = image_dir

    def __getitem__(self, index):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        ann_ids = self.coco.getAnnIds(imgIds=image_id) # list
        image_infos = self.coco.loadImgs(image_id)[0]
        anns = self.coco.loadAnns(ann_ids)

        # cv2를 활용하여 image 불러오기
        file_name = image_infos["file_name"]
        images = cv2.imread(os.path.join(self.image_dir, file_name))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            cat_vector = np.zeros((class_num, ))
            for i in range(len(anns)):
                cat_id = make_cls_id(anns[i]['category_id'])
                cat_vector[cat_id] = 1

            return images, cat_vector
        elif self.mode == "test":
            # transform -> albumentations
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images
        else:
            raise RuntimeError("CustomDataLoader mode error")

    def __len__(self):
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

# def collate_fn(batch):
#     return tuple(zip(*batch))

train_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])


# if __name__ == '__main__':
#     train_dataset = CustomDataLoader(
#     data_dir=train_path, mode="train", transform=train_transform, class_num=class_num)

#     batch_size = 2 
#     workers = 1
#     use_cuda = torch.cuda.is_available()

#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size,
#         num_workers=workers,
#         shuffle=False,
#         pin_memory=use_cuda,
#         # collate_fn=collate_fn,
#         drop_last=True,
#     )

#     images, labels = next(iter(train_loader))
#     print(images.shape, labels.shape)
#     print(labels.type())
#     labels = labels.type(torch.FloatTensor)
#     print(labels.type())