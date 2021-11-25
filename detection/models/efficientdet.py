import torch
import torch.nn as nn
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet


# https://github.com/rwightman/efficientdet-pytorch
def get_net():
    config = get_efficientdet_config('tf_efficientdet_d5')

    net = EfficientDet(config, pretrained_backbone=False)

    checkpoint = torch.load('../saved_models/efficientdet_d5-ef44aea8.pth')
    net.load_state_dict(checkpoint)
    net.reset_head(num_classes=39)

    # 이 부분 수정 필요 
    # config.num_classes = 39
    # config.image_size = 1500

    net.class_net = HeadNet(config, num_outputs=39)
    #return DetBenchTrain(net, config)
    return net, config

class efficientdet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.config = get_net()
        
        
    def forward(self, x):
        return self.model(x)

