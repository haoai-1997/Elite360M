import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder_util import UpSampleBN, norm_normalize


class depth_head(nn.Module):
    def __init__(self, features, scale):
        super(depth_head, self).__init__()
        features = int(features)
        self.depth_output = nn.Conv2d(features, 1, kernel_size=3, stride=1, padding=1)
        self.scale = scale

    def forward(self, fused_feature):
        x = F.interpolate(fused_feature, scale_factor=self.scale, mode='bilinear')
        depth_pred = self.depth_output(x)
        return depth_pred


class rgb_head(nn.Module):
    def __init__(self, features, scale):
        super(rgb_head, self).__init__()
        features = int(features)
        self.rgb_output = nn.Conv2d(features, 3, kernel_size=3, stride=1, padding=1)
        self.scale = scale

    def forward(self, fused_feature):
        x = F.interpolate(fused_feature, scale_factor=self.scale, mode='bilinear')
        rgb_pred = self.rgb_output(x)
        return rgb_pred


class segmentation_head(nn.Module):
    def __init__(self, features, num_classes, scale):
        super(segmentation_head, self).__init__()
        features = int(features)
        self.num_classes = num_classes
        self.scale = scale
        self.linear_pred = nn.Conv2d(features, self.num_classes * 3 * 3, kernel_size=1)
        self.DAP = nn.Sequential(
            nn.PixelShuffle(3),
            nn.AvgPool2d((3, 3))
        )

    def forward(self, fused_feature):
        x = F.interpolate(fused_feature, scale_factor=self.scale, mode='bilinear')
        seg_pred = self.DAP(self.linear_pred(x.contiguous()))

        return seg_pred


class normal_head(nn.Module):
    def __init__(self, features, scale):
        super(normal_head, self).__init__()
        features = int(features)
        self.normal_output = nn.Conv2d(features, 3, kernel_size=3, stride=1, padding=1)

        self.scale = scale

    def forward(self, fused_feature):
        normal_ = F.interpolate(fused_feature, scale_factor=self.scale, mode='bilinear')
        normal_pred = self.normal_output(normal_)
        normal_pred = norm_normalize(normal_pred)
        return normal_pred


class shared_decoder_backbone(nn.Module):
    def __init__(self, channel, backbone, output_dim, features):
        super(shared_decoder_backbone, self).__init__()
        features = int(features)
        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        if backbone == 'res18' or backbone == 'res34':
            self.up1 = UpSampleBN(skip_input=features // 1 + 256, output_features=features // 2)
        elif backbone == 'res50':
            self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2)
        elif backbone == 'eff-b5':
            self.up1 = UpSampleBN(skip_input=features // 1 + 176, output_features=features // 2)
        elif backbone == 'swin-b':
            self.up1 = UpSampleBN(skip_input=features // 1 + 512, output_features=features // 2)
        self.out = nn.Conv2d(features // 2, output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, shared_representation, erp_feature):

        x_d0 = self.conv2(shared_representation.contiguous())

        x_d1 = self.up1(x_d0, erp_feature)

        output = self.out(x_d1)

        return output
