import torch
import torch.nn as nn
from torchvision import models


class ResNet101(nn.Module):
    def __init__(self, num_classes, cls_classes = None, device = 'cuda'):
        super().__init__()

        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        output = self.backbone(x)
        output = self.sigmoid(output)
        return output, 0

    def get_loss(self, outs, cls_outs, labels, criterion):
        labels = labels.type(torch.FloatTensor).to(self.device)
        return criterion(outs, labels)


# resnet for dependency, do not load in training session
class ResNet101_(nn.Module):
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

        self.feature_model = ResNet101_(num_classes)
        self.cls_model = ResNet101_(cls_classes)
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
        label_binary = self.get_binary_label(labels).to(torch.long) # .to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, torch.max(label, dim=1)[1])
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))

    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape))
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)


class multihead_hooked(nn.Module):
    def __init__(self, num_classes, cls_classes, device):
        super().__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fcs = nn.ModuleList([nn.Linear(2048, 2) for _ in range(38)])
        self.device = device

        #option for gradcam
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.OODhook = []
        
        self.layerhook.append(self.backbone.layer4.register_forward_hook(self.forward_hook()))

        #option for OOD detection
        self.backbone.conv1.register_forward_hook(self.dense_hook())

    
    def activations_hook(self,grad):
        self.gradients = grad


    def get_act_grads(self):
        return self.gradients


    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook


    def dense_hook(self):
        def hook(model, input, output):
            self.OODhook.append(output.detach())
        return hook


    def forward(self, inputs):
        feat = self.backbone(inputs)
        vecs = []
        for fc in self.fcs:
            vec = fc(feat)
            vecs.append(vec)
        
        stack = torch.stack(vecs, axis = 0)
        return stack.permute(1,0,2), 0 #self.selected_out


    def get_loss(self, outs, cls_outs, labels, criterion):
        label_binary = self.get_binary_label(labels).to(torch.long) # .to(torch.float32)
        losses = []
        for out, label in zip(outs, label_binary):
            _loss = criterion(out, torch.max(label, dim=1)[1])
            losses.append(_loss)
        
        return torch.sum(torch.stack(losses))


    def get_binary_label(self, labels):
        _ones = torch.ones((labels.shape)).to('cuda')
        counterpart = _ones - labels
        cats = torch.stack([counterpart, labels], axis = 0)
        return cats.permute(1,2,0).to(self.device)