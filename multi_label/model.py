import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet101(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.backbone(x)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output





