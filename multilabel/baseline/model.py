import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)
        # self.fc1 = nn.Linear(1000, 500)
        # self.fc2 = nn.Linear(500, 250)
        # self.fc3 = nn.Linear(250, 125)
        # self.fc4 = nn.Linear(125, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.backbone(x)
        # output = self.fc1(output)
        # output = self.fc2(output)
        # output = self.fc3(output)
        # output = self.fc4(output)
        output = self.sigmoid(output)
        return output


