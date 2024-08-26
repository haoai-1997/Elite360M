import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# normalize
def norm_normalize(norm_out):
    norm_x, norm_y, norm_z = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=1)
    return final_out


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class UpSampleBN_task(nn.Module):
    def __init__(self, task_input, unified_input, output_features):
        super(UpSampleBN_task, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(task_input + unified_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU())

    def forward(self, task_feature, unified_feature):
        up_x = F.interpolate(task_feature, size=[unified_feature.size(2), unified_feature.size(3)], mode='bilinear',
                             align_corners=True)
        f = torch.cat([up_x, unified_feature], dim=1)
        return self._net(f)


class MultiHead_Self_Attention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, input_features):
        super().__init__()

        self.n_head = n_head
        self.d_k = input_features // n_head

        self.w_q = nn.Linear(input_features, input_features)
        self.w_k = nn.Linear(input_features, input_features)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fea):
        d_k, n_head = self.d_k, self.n_head

        sz_b, len_q, _ = fea.size()

        q = self.w_q(fea).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(fea).view(sz_b, len_q, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lk x dk

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / np.power(self.d_k, 0.5)
        attn = self.softmax(attn)

        return attn


class Cross_Task_Attention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, input_features):
        super().__init__()

        self.n_head = n_head
        self.d_k = input_features // n_head
        self.d_v = input_features // n_head

        self.w_v = nn.Linear(input_features, input_features)
        self.mlp1 = nn.Linear(input_features, input_features)
        self.mlp2 = nn.Linear(input_features, input_features // 2)
        self.mlp3 = nn.Linear(input_features, input_features // 2)
        self.mlp = nn.Linear(input_features * 2, input_features)

    def forward(self, attn1, attn2, attn3, v):
        sz_b, len_v, _ = v.size()
        d_v, n_head = self.d_v, self.n_head

        v = self.w_v(v).view(sz_b, len_v, n_head, d_v)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        A_1 = torch.bmm(attn1, v)

        A_1 = A_1.view(n_head, sz_b, len_v, d_v)
        A_1 = (A_1.permute(1, 2, 0, 3).contiguous().view(sz_b, len_v, -1))

        A_2 = torch.bmm(attn2, v)

        A_2 = A_2.view(n_head, sz_b, len_v, d_v)
        A_2 = (A_2.permute(1, 2, 0, 3).contiguous().view(sz_b, len_v, -1))

        A_3 = torch.bmm(attn3, v)

        A_3 = A_3.view(n_head, sz_b, len_v, d_v)
        A_3 = (A_3.permute(1, 2, 0, 3).contiguous().view(sz_b, len_v, -1))
        A_1 = self.mlp1(A_1)
        A_2 = self.mlp2(A_2)
        A_3 = self.mlp3(A_3)
        concat_A = self.mlp(torch.cat([A_1, A_2, A_3], dim=-1)) + A_1
        return concat_A


class Triplet_cross_attention(nn.Module):
    def __init__(self, input_dim, head):
        super(Triplet_cross_attention, self).__init__()

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.msa1 = MultiHead_Self_Attention(head, input_dim)
        self.msa2 = MultiHead_Self_Attention(head, input_dim)
        self.msa3 = MultiHead_Self_Attention(head, input_dim)

        self.cross_task_attention1 = Cross_Task_Attention(head, input_dim)
        self.cross_task_attention2 = Cross_Task_Attention(head, input_dim)
        self.cross_task_attention3 = Cross_Task_Attention(head, input_dim)

    def forward(self, feature_1, feature_2, feature_3):
        B, _, H, W = feature_1.shape
        feature1_flatten = feature_1.flatten(-2).permute(0, 2, 1)
        feature2_flatten = feature_2.flatten(-2).permute(0, 2, 1)
        feature3_flatten = feature_3.flatten(-2).permute(0, 2, 1)

        feature1_flatten_norm = self.norm1(feature1_flatten)
        feature2_flatten_norm = self.norm2(feature2_flatten)
        feature3_flatten_norm = self.norm3(feature3_flatten)

        attention_map_1 = self.msa1(feature1_flatten_norm)
        attention_map_2 = self.msa2(feature2_flatten_norm)
        attention_map_3 = self.msa3(feature3_flatten_norm)

        feature1_output = self.cross_task_attention1(attention_map_1, attention_map_2, attention_map_3,
                                                     feature1_flatten_norm)

        feature2_output = self.cross_task_attention2(attention_map_2, attention_map_1, attention_map_3,
                                                     feature2_flatten_norm)

        feature3_output = self.cross_task_attention3(attention_map_3, attention_map_1, attention_map_2,
                                                     feature3_flatten_norm)

        output_1 = rearrange(feature1_output, 'b (h w) c-> b c h w', h=H)
        output_2 = rearrange(feature2_output, 'b (h w) c-> b c h w', h=H)
        output_3 = rearrange(feature3_output, 'b (h w) c-> b c h w', h=H)

        return output_1, output_2, output_3
