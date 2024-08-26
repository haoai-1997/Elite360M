import torch.nn as nn
from .swin_pytorch.Swin_Transformer import SwinTransformer


class SwinB(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinB, self).__init__()
        # compute healpixel
        self.vit_encoder = SwinTransformer(pretrain_img_size=224,
                                           embed_dim=128,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[4, 8, 16, 32],
                                           window_size=7,
                                           drop_path_rate=0.5,
                                           frozen_stages=-1)
        if pretrained:
            self.init_weights("checkpoints/swin_base_patch4_window7_224_22k.pth")

    def init_weights(self, pretrained_model):
        self.vit_encoder.init_weights(pretrained_model)

    def forward(self, x):
        out = self.vit_encoder(x)
        return out


class SwinT(nn.Module):
    def __init__(self, pretrained=False):
        super(SwinT, self).__init__()
        # compute healpixel
        self.vit_encoder = SwinTransformer(pretrain_img_size=224,
                                           embed_dim=96,
                                           depths=[2, 2, 6, 2],
                                           num_heads=[3, 6, 12, 24],
                                           window_size=7,
                                           drop_path_rate=0.5,
                                           frozen_stages=-1)
        if pretrained:
            self.init_weights("checkpoints/swin_tiny_patch4_window7_224_22k.pth")

    def init_weights(self, pretrained_model):
        self.vit_encoder.init_weights(pretrained_model)

    def forward(self, x):
        out = self.vit_encoder(x)
        return out


class SwinL(nn.Module):
    def __init__(self, pretrained=False):
        super(SwinL, self).__init__()
        # compute healpixel
        self.vit_encoder = SwinTransformer(pretrain_img_size=224,
                                           embed_dim=192,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[3, 6, 12, 24],
                                           window_size=7,
                                           drop_path_rate=0.5,
                                           frozen_stages=-1)
        if pretrained:
            self.init_weights("checkpoints/swin_large_patch4_window7_224_22k.pth")

    def init_weights(self, pretrained_model):
        self.vit_encoder.init_weights(pretrained_model)

    def forward(self, x):
        out = self.vit_encoder(x)
        return out
