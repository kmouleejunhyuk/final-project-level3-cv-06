import albumentations as A
import albumentations.augmentations.crops.transforms as T
import numpy as np
import cv2


class processer():
    def __init__(self, bar_margin = 75, side_margin = 150):
        self.formal_transform = A.Compose(
            [
                T.Crop(0, 0, 1920, 1080 - bar_margin, always_apply=True),
                T.CenterCrop(1080 - bar_margin, 1920 - (side_margin * 2), always_apply=True),
            ]
        )

    # 많지 않은 샘플이기에, 속도가 많이 느릴것 같지 않아 numba 제외. 
    # 필요하면 @njit 데코레이터만 넣으면 됨
    def get_box(self, image: np.ndarray):
        '''
        이미지를 순회하면서 객체가 들어 있는 가장 큰 box 좌표를 구해주는 코드
        Args: 
            image: np.ndarray
                blur 처리된 image
        returns:
            list of box coordinates([xmin, ymin, xmax, ymax])
        '''
        xmin, ymin, xmax, ymax = 1e+5, 1e+5, 0, 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pix = sum(image[i, j, :]) / 3

                if pix < 220:
                    if i < ymin: 
                        ymin = i
                    if j < xmin:
                        xmin = j
                    if i > ymax:
                        ymax = i
                    if j > xmax:
                        xmax = j
        
        return [xmin, ymin, xmax, ymax]


    def cropper(self, image):
        '''
        image 배경을 제거하는 코드
        Args:
            image: np.ndarray
        return
            c_image: torch.Tensor
                transformed image tensor
        '''
        meta = self.formal_transform(image = image)
        image = meta['image']

        b_image = cv2.GaussianBlur(image, ksize=(7,7), sigmaX=0)
        xmin, ymin, xmax, ymax = list(map(int, self.get_box(b_image)))
        post_transform = A.Compose([
            T.Crop(xmin, ymin, xmax, ymax, always_apply=True)
        ])

        meta = post_transform(image = image)
        c_image = meta['image']
        return c_image


    def preprocess(self, image: np.ndarray):
        if image.shape == (1080, 1920, 3):
            image = self.cropper(image)
        else:
            print('not a proper image. preprocess skipped')

        return image

        
    



        


    
