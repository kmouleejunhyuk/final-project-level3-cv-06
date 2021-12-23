import torch

from pytorch_lightning import LightningModule
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detection.config.detection_config import config as CONFIG

class DetectionModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_classes = CONFIG.num_classes 
        
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(self, imgs):
        self.model.eval()
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.lr)
        return optimizer

