import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def read_img(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    return img


class CustomDataset(Dataset):
    '''
    COCO format dataset
        data_dir : data path
        transforms : data transform (resize, crop, ToTensor, etc...)
    '''
    def  __init__(self, annotation, data_dir, mode='train', transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.mode = mode
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image = read_img(self.data_dir+image_info['path']+image_info['file_name'])

        if self.mode in ('train', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
            anns = self.coco.loadAnns(ann_ids)

            boxes = np.array([x['bbox'] for x in anns])

            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # labels
            labels = np.array([x['category_id']+1 for x in anns])
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # areas
            areas = np.array([x['area'] for x in anns])
            areas = torch.as_tensor(areas, dtype=torch.float32)

            target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

            # transform
            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']
                target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            
            return image, target
        
        elif self.mode == 'test':
            # transform
            if self.transforms:
                sample = {
                    'image':image
                }
                sample = self.transforms(**sample)
                image = sample['image']

            return image
        
        else:
            RuntimeError("CustomDataset mode error!")
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())


# def train_transform():
#     return A.Compose([
#         ToTensorV2()
#     ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})
def train_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=1500, min_width=1500, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})

# def valid_transform():
#     return A.Compose([
#         ToTensorV2()
#     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
def valid_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=1500, min_width=1500, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})
