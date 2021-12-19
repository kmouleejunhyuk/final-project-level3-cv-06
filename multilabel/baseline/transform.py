from albumentations.pytorch import ToTensorV2
from imgaug.augmenters.size import pad
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A


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


val_transform = A.Compose([
    ratio_aware_pad(),
    A.Resize(512, 512),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])

train_transform = A.Compose([
    ratio_aware_pad(),
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

train_aug_transform = A.Compose([
    ratio_aware_pad(),
    A.Resize(512, 512),
    A.OneOf([
        A.CLAHE(always_apply=False, p=0.5, clip_limit=(6, 18), tile_grid_size=(20, 18)),
        A.Sharpen(p=0.5)
    ], p=1),
    A.Normalize(),
    ToTensorV2()
])

test_transform = A.Compose([
    ratio_aware_pad(),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ToTensorV2()
])
