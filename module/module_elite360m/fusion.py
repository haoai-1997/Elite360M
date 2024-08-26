import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class SA_affinity_attention(nn.Module):

    def __init__(self, ico_dim, erp_dim, iteration, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.ico_dim = ico_dim
        self.erp_dim = erp_dim
        self.iters = iteration
        self.scale = qk_scale or erp_dim ** -0.5
        self.q_linear = nn.ModuleList()
        for i in range(self.iters):
            q_linear = nn.Linear(erp_dim, erp_dim, bias=False)
            self.q_linear.append(q_linear)

        self.kv_linear = nn.ModuleList()
        for i in range(self.iters):
            kv_linear = nn.Linear(ico_dim, erp_dim * 2, bias=False)
            self.kv_linear.append(kv_linear)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(erp_dim, erp_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        for i in range(self.iters):
            q_ = self.q_linear[i](q)
            kv_ = self.kv_linear[i](kv)

            B_q, N_q, C_q = q_.shape
            B_kv, N_kv, C_kv = kv_.shape

            kv_ = kv_.reshape(B_kv, -1, 2, C_kv // 2).permute(2, 0, 1, 3)
            k, v = kv_[0], kv_[1]

            attn = (q_ @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            q = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)

        x = self.proj(q)
        x = self.proj_drop(x)

        return x


class DA_affinity_attention(nn.Module):
    def __init__(self, ico_dim, erp_dim, iteration, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.ico_dim = ico_dim
        self.erp_dim = erp_dim
        self.iters = iteration
        self.scale = qk_scale or erp_dim ** -1
        self.q_linear = nn.ModuleList()
        for i in range(self.iters):
            q_linear = nn.Linear(erp_dim, erp_dim, bias=False)
            self.q_linear.append(q_linear)

        self.kv_linear = nn.ModuleList()
        for i in range(self.iters):
            kv_linear = nn.Linear(ico_dim, erp_dim * 2, bias=False)
            self.kv_linear.append(kv_linear)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(erp_dim, erp_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fc_delta = nn.ModuleList()
        for i in range(self.iters):
            fc_delta = nn.Linear(3, erp_dim, bias=False)
            self.fc_delta.append(fc_delta)

    def forward(self, q, q_coord, kv, kv_coord):
        for i in range(self.iters):
            q_ = self.q_linear[i](q)
            kv_ = self.kv_linear[i](kv)

            B_kv, N_kv, C_kv = kv_.shape

            kv_ = kv_.reshape(B_kv, -1, 2, C_kv // 2).permute(2, 0, 1, 3)
            k, v = kv_[0], kv_[1]

            pos_enc = self.fc_delta[i](torch.exp(-torch.abs(q_coord[:, :, None] - kv_coord[:, None])))

            attn = torch.exp(-torch.abs(q_[:, :, None] - k[:, None])) + pos_enc

            attn = F.softmax(attn.sum(-1) * self.scale, dim=-1)

            attn = self.attn_drop(attn)

            q = (attn @ v)

        x = self.proj(q)
        x = self.proj_drop(x)

        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class EI_Adaptive_Fusion(nn.Module):
    def __init__(self, resolution, embedding_dim):
        super(EI_Adaptive_Fusion, self).__init__()
        self.resolution = resolution

        self.embed_channel = embedding_dim

        self.SA_attn_bottle = SA_affinity_attention(ico_dim=self.embed_channel, erp_dim=self.embed_channel, iteration=1)

        self.DA_attn_bottle = DA_affinity_attention(ico_dim=self.embed_channel, erp_dim=self.embed_channel, iteration=1)

        self.gate_SA_bottle = nn.Linear(self.embed_channel * 2, self.embed_channel, bias=False)

        self.gate_DA_bottle = nn.Linear(self.embed_channel * 2, self.embed_channel, bias=False)

        self.sigmoid = nn.Sigmoid()

    def calculate_erp_coord(self, erp_feature):
        def coords2uv(coords, w, h, fov=None):
            # output uv size w*h*2
            uv = torch.zeros_like(coords, dtype=torch.float32)
            middleX = w / 2 + 0.5
            middleY = h / 2 + 0.5
            if fov == None:
                uv[..., 0] = (coords[..., 0] - middleX) / w * 2 * np.pi
                uv[..., 1] = (coords[..., 1] - middleY) / h * np.pi
            else:
                fov_h, fov_w = pair(fov)
                uv[..., 0] = (coords[..., 0] - middleX) / w * (fov_w / 360) * 2 * np.pi
                uv[..., 1] = (coords[..., 1] - middleY) / h * (fov_h / 180) * np.pi
            return uv

        def uv2xyz(uv):
            sin_u = torch.sin(uv[..., 0])
            cos_u = torch.cos(uv[..., 0])
            sin_v = torch.sin(uv[..., 1])
            cos_v = torch.cos(uv[..., 1])
            return torch.stack([
                cos_v * sin_u,
                sin_v,
                cos_v * cos_u,
            ], dim=-1)

        if len(erp_feature.shape) == 4:
            bs, channel, h, w = erp_feature.shape
        else:
            bs, hxw, channel = erp_feature.shape
            h = int(np.sqrt(hxw // 2))
            w = h * 2
        erp_yy, erp_xx = torch.meshgrid(torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w), indexing='ij')
        screen_points = torch.stack([erp_xx, erp_yy], -1)
        erp_coordinate = uv2xyz(coords2uv(screen_points, w, h))
        erp_coordinate = erp_coordinate[None, ...].repeat(bs, 1, 1, 1).to(erp_feature.device)

        erp_coordinate = rearrange(erp_coordinate, "n h w c-> n (h w) c")
        return erp_coordinate

    def forward(self, erp_feature, ico_feature, ico_coord):
        erp_coordinate = self.calculate_erp_coord(erp_feature)

        bs, c, h, w = erp_feature.shape
        erp_feature = rearrange(erp_feature, 'n c h w -> n (h w) c')

        ## ICO_process

        sa_fusion_feature = self.SA_attn_bottle(erp_feature, ico_feature)

        da_fusion_feature = self.DA_attn_bottle(erp_feature, erp_coordinate, ico_feature, ico_coord)

        sa_factor = self.sigmoid(self.gate_SA_bottle(torch.cat([sa_fusion_feature, da_fusion_feature], dim=-1)))

        da_factor = self.sigmoid(self.gate_DA_bottle(torch.cat([sa_fusion_feature, da_fusion_feature], dim=-1)))

        fusion_feature = sa_factor * sa_fusion_feature + da_factor * da_fusion_feature

        fusion_feature = rearrange(fusion_feature, "n (h w) c-> n c h w", h=h)

        return fusion_feature
