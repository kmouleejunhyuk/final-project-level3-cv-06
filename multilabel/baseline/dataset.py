import os
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import cv2
import torch

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from imgaug.augmenters.size import pad


category_names = ['Aerosol', 'Alcohol', 'Awl', 'Axe', 'Bat', 'Battery', 'Bullet', 'Firecracker', 'Gun', 'GunParts', 'Hammer',
                  'HandCuffs', 'HDD', 'Knife', 'Laptop', 'Lighter', 'Liquid', 'Match', 'MetalPipe', 'NailClippers', 'PortableGas', 'Saw', 'Scissors', 'Screwdriver',
                  'SmartPhone', 'SolidFuel', 'Spanner', 'SSD', 'SupplymentaryBattery', 'TabletPC', 'Thinner', 'USB', 'ZippoOil', 'Plier', 'Chisel', 'Electronic cigarettes',
                  'Electronic cigarettes(Liquid)', 'Throwing Knife']

class_num = 38


# def make_cls_id(origin_id):
#     # 14, 35, 40 label이 없음
#     if 0 < origin_id < 14:
#         cat_id = origin_id - 1
#         return cat_id

#     elif 15 <= origin_id < 35:
#         cat_id = origin_id - 2
#         return cat_id

#     elif 36 <= origin_id < 40:
#         cat_id = origin_id - 3
#         return cat_id
#     elif origin_id == 41:
#         cat_id = origin_id - 4
#         return cat_id
#     else:
#         raise Exception(f'없는 category id 입니다 {origin_id}')


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
        ann_ids = self.coco.getAnnIds(imgIds=image_id)  # list
        image_infos = self.coco.loadImgs(image_id)[0]
        anns = self.coco.loadAnns(ann_ids)

        # cv2를 활용하여 image 불러오기
        file_path = image_infos["path"]
        file_name = image_infos["file_name"]

        path = os.path.join(self.image_dir, file_path[1:], file_name)

        images = cv2.imread(path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            cat_vector = np.zeros((class_num, ))
            for i in range(len(anns)):
                # cat_id = make_cls_id(anns[i]['category_id'])
                cat_id = anns[i]['category_id']
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


class ratio_aware_pad(ImageOnlyTransform):
    def __init__(self, padmax = None):
        #usage: 
        #  A.Compose([ratio_aware_pad(padmax = 2000)])
        #  maximum size pad(output: 2000 * 2000 * 3), may be sparse

        #usage2: 
        #   A.Compose([ratio_aware_pad(), A.Resize(512, 512)])
        #   ratio aware pad, less sparse, must be resized(size may differ by image)

        super().__init__(always_apply=True)
        self.padmax = padmax

    def apply(self, img, **params):
        if self.padmax:
            #max size aware padding
            #크기 이상인 이미지는 없다고 가정
            assert img.shape[0] < self.padmax and img.shape[1] < self.padmax
            u, r = (self.padmax - img.shape[0]) // 2, (self.padmax - img.shape[1]) // 2
            d, l = self.padmax - img.shape[0] - u, self.padmax - img.shape[1] - r
            img = pad(img, top = u, bottom = d, right = r, left = l, cval = 250)
            
        else:   #ratio-aware padding
            h, w = img.shape[0], img.shape[1]
            if h == w: return img

            if h > w:
                r_delta = (h - w) // 2
                l_delta = (h - w) - r_delta
                img = pad(img, right = r_delta, left = l_delta, cval = 250)
            else:
                u_delta = (w - h) // 2
                d_delta = (w - h) - u_delta
                img = pad(img, top = u_delta, bottom = d_delta, cval = 250)

            assert img.shape[0] == img.shape[1]
        return img


train_transform = A.Compose([
    ratio_aware_pad(),
    A.augmentations.geometric.resize.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])

val_transform = A.Compose([
    ratio_aware_pad(),
    A.augmentations.geometric.resize.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])

test_transform = A.Compose([
    ratio_aware_pad(),
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
