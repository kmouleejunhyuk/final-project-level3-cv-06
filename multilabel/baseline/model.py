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

class Twostage(nn.Module):
    def __init__(self, num_classes, cls_classes, device):
        super().__init__()

        self.feature_model = ResNet101(num_classes)
        self.cls_model = ResNet101(cls_classes)
        self.device = device

    def forward(self, x):
        out1 = self.feature_model(x)
        cls_out = self.cls_model(x)
        return out1, cls_out

    def get_loss(self, outs, cls_outs, labels, criterion):
        labels, cls_labels = self.get_cls_labels(labels)
        feat_loss = criterion(outs, labels)
        cls_loss = criterion(cls_outs, cls_labels)
        
        return feat_loss + cls_loss

    def get_cls_labels(self, labels):
        labels = labels.type(torch.FloatTensor)
        cls_labels = torch.clamp(torch.sum(labels, dim=-1), min = 0, max = 5)
        cls_labels = torch.nn.functional.one_hot(cls_labels.to(torch.int64), 6).float()

        return labels.to(self.device), cls_labels.to(self.device)

