import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class FCNN(nn.Module):
    def __init__(self):
        num_classes = 39 # include background (0: background)

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        self.backbone.out_channels = 1280

        self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                aspect_ratios=((0.5, 1.0, 2.0),))
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=7,
                                                             sampling_ratio=2)
        self.model = FasterRCNN(backbone=self.backbone,
                                num_classes=num_classes,
                                rpn_anchor_generator=self.anchor_generator,
                                box_roi_pool=self.roi_pooler)
    
    def forward(self, x):
        return self.model(x)
