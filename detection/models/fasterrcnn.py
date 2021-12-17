import torch

from pytorch_lightning import LightningModule
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class DetectionModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_classes = 39 # include background (0: background)
        
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(self, imgs):
        self.model.eval()
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
