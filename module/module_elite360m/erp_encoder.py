import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from backbone.efficientnet import EfficientNetB0, EfficientNetB5
from backbone.swin_transformer import SwinB, SwinT, SwinL
from backbone.dilateformer import Dilateformer_T

Encoder = {
           'res18': resnet18,
           'res34': resnet34,
           'res50': resnet50,
           'eff-b5': EfficientNetB5,
           'swin-b': SwinB,
           'swin-t': SwinT,
           'dilate-t': Dilateformer_T
           }

class ERP_encoder_Res(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_Res, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        encoder = pretrained_model
        # ResNet34
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512
        if int(backbone[-2:]) >= 50:
            self.down = nn.Conv2d(512 * 4, channel, kernel_size=1, stride=1, padding=0)
        else:
            self.down = nn.Conv2d(512, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        conv1 = self.relu(self.bn1(self.conv1(erp_rgb)))  # h/2 * w/2 * 64
        pool = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        x_downsample = []
        layer1 = self.layer1(pool)  # h/4 * w/4 * 64
        layer2 = self.layer2(layer1)  # h/8 * w/8 * 128
        layer3 = self.layer3(layer2)  # h/16 * w/16 * 256
        layer4 = self.layer4(layer3)  # h/32 * w/32 * 512
        layer4_reshape = self.down(layer4)  # h/32 * w/32 * embedding channel
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        return layer4_reshape, x_downsample