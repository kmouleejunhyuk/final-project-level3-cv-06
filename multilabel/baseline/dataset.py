import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import re


class CustomDataset(Dataset):
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

        path = os.path.join(self.image_dir, self.mode, file_path[1:], file_name)

        images = cv2.imread(path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if self.mode in ("train", "eval", "sampled"):
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            cat_vector = np.zeros((self.class_num, ))
            for i in range(len(anns)):
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


class RetrainDataset(Dataset):
    """
    custom format(file + filename)
    
    """
    def __init__(self, image_dirs: list, mode="train", transform=None, class_num=38):
        '''
        dataset for retrain in custom format
        Args:
            image_dirs: list[str]
                이미지 경로들 리스트
                ex: ['static/ml_pred/000001[1,23,45].png', 'static/ml_pred/000001[1,23,45].png' , ...]

            mode: str
                dummy var for compatibility

            transform: Albumentation.Compose
                Albumentation Compose object
                ex: Albumentation.Compose([Albumentation.TotensorV2()])

            class_num: int
        '''
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.class_num = class_num
        self.image_dir = image_dirs

    def __getitem__(self, index):
        imgpath = self.image_dir[index]
        labels = np.zeros((self.class_num, ))

        index = self.get_label_from_dir(imgpath)
        labels[index] = 1

        images = cv2.imread(imgpath)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]
        
        return images, labels


    def __len__(self):
        return len(self.image_dir)

    def get_label_from_dir(self, strpath: str):
        '''
        get labels from filename(used for custom retraining session)
        args:
            strpath: 단일 이미지 경로
                ex: static/ml_pred/000001[1,23,45].png

        return:
            index: list[int] with label index
                ex: [1, 23, 45]
        '''
        regex = r"[^[]*\[([^]]*)\]"
        filename = strpath.split(os.sep)[-1]
        parsed = re.match(regex, filename).groups()[0]
        index = list(map(int, parsed.split(',')))
        return index



    

    