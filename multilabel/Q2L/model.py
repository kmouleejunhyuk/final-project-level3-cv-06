import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils.misc import clean_state_dict
from models.query2label import build_q2l
import os
import yaml
import addict


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


class Q2L(nn.Module):
    def __init__(self, num_classes, from_pretrained = False):
        super().__init__()

        print('loading yaml for Q2L', end="\r")
        with open('/opt/ml/finalproject/multilabel/Q2L/configs/Q2L.yaml') as f:
            config_train = yaml.load(f, Loader=yaml.FullLoader)
        args = addict.Dict(config_train)
        print('yaml loading complete        ')

        model = build_q2l(args)
        # ema_m = ModelEma(model, args.ema_decay) # 0.9997
        # exponential moving average. for data imbalance treatment. 
        # not implmenting in our code(leaving for future)
        if from_pretrained:
            dir = '/opt/ml/finalproject/multilabel/Q2L/weights' + '/CvT-w24-384x384-IN-22k.pkl'
            if os.path.isfile(dir):
                print("=> loading checkpoint from '{}'".format(dir), end = '\r')
                checkpoint = torch.load(dir, map_location=torch.device('cuda'))

                if 'state_dict' in checkpoint:
                    state_dict = clean_state_dict(checkpoint['state_dict'])
                # elif 'model' in checkpoint:
                #     state_dict = clean_state_dict(checkpoint['model'])
                else:
                    raise ValueError("No model or state_dict Found!!!")

                model.load_state_dict(state_dict, strict=False)
                print("=> loaded checkpoint      ")
                if model.fc.num_class != num_classes:
                    past = model.fc.num_class
                    model.reset_fc(num_classes)
                    print(f'Changed fc output from {past} to {model.fc[-1].out_features}')
                del checkpoint
                del state_dict
                torch.cuda.empty_cache() 
            else:
                NotImplementedError("=> no checkpoint found at '{}'".format(dir))

        self.model = model

    def forward(self, input):
        return self.model(input)



