import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.backbone(x)
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


class multihead(nn.Module):
    def __init__(self, num_classes, cls_classes, device):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fcs = nn.ModuleList([nn.Linear(2048, 2) for _ in range(38)])
        self.device = device

    def forward(self, inputs):
        feat = self.backbone(inputs)
        vecs = []
        for fc in self.fcs:
            vec = fc(feat)
            vecs.append(vec)
        
        stack = torch.stack(vecs, axis = 0)
        return stack.permute(1,0,2), 0


    def get_loss(self, outs, cls_outs, labels, criterion):
        label_binary = self.get_binary_label(labels).to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, label)
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))

    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape))
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)


class multihead_with_seq(nn.Module):
    def __init__(self, num_classes, cls_classes, device):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.device = device

        mlist = []
        for _ in range(38):
            if _ in [1, 30]:
                seq = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                )
                mlist.append(seq)
            else:
                mlist.append(nn.Linear(2048, 2))

        self.fcs = nn.ModuleList(mlist)
    def forward(self, inputs):
        feat = self.backbone(inputs)
        vecs = []
        for fc in self.fcs:
            vec = fc(feat)
            vecs.append(vec)
        
        stack = torch.stack(vecs, axis = 0)
        return stack.permute(1,0,2), 0


    def get_loss(self, outs, cls_outs, labels, criterion):
        label_binary = self.get_binary_label(labels).to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, label)
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))

    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape))
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)


class multihead_with_quant(nn.Module):
    def __init__(self, num_classes, cls_classes, device):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fcs = nn.ModuleList([nn.Linear(2048, 2) for _ in range(num_classes)])
        self.device = device
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.quant(inputs)
        feat = self.backbone(inputs)
        vecs = []
        for fc in self.fcs:
            vec = fc(feat)
            vecs.append(vec)
        
        stack = torch.stack(vecs, axis = 0)
        stack = stack.permute(1,0,2)
        out = self.dequant(stack)
        return self.sigmoid(out), 0


    def get_loss(self, outs, cls_outs, labels, criterion):
        label_binary = self.get_binary_label(labels).to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, label)
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))

    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape))
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)