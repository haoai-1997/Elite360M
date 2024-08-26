import math
import torch
from torch import nn
from torch.nn import functional as F
from .efficientnet_pytorch.model import EfficientNet as effNet

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0, self).__init__()

        # load pretrained EfficientNet B3
        if pretrained == True:
            self.model_ft = effNet.from_pretrained('efficientnet-b0',
                                                   weights_path="checkpoints/adv-efficientnet-b0-b64d5a18.pth")
        else:
            self.model_ft = effNet.from_name('efficientnet-b0')
        del self.model_ft._conv_head
        del self.model_ft._bn1
        del self.model_ft._fc

    def forward(self, x):
        endpoints = self.model_ft.extract_endpoints(x)
        return endpoints
class EfficientNetB1(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB1, self).__init__()

        # load pretrained EfficientNet B3
        if pretrained == True:
            self.model_ft = effNet.from_pretrained('efficientnet-b1',
                                                   weights_path="checkpoints/adv-efficientnet-b1-f1951068.pth")
        else:
            self.model_ft = effNet.from_name('efficientnet-b1')
        del self.model_ft._conv_head
        del self.model_ft._bn1
        del self.model_ft._fc

    def forward(self, x):
        endpoints = self.model_ft.extract_endpoints(x)
        return endpoints
class EfficientNetB5(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB5, self).__init__()

        # load pretrained EfficientNet B3
        if pretrained == True:
            self.model_ft = effNet.from_pretrained('efficientnet-b5',
                                                   weights_path="checkpoints/adv-efficientnet-b5-86493f6b.pth")
        else:
            self.model_ft = effNet.from_name('efficientnet-b5')
        del self.model_ft._conv_head
        del self.model_ft._bn1
        del self.model_ft._fc

    def forward(self, x):
        endpoints = self.model_ft.extract_endpoints(x)
        return endpoints
