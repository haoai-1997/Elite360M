import os
import torch
import torch.nn as nn
from module.module_elite360m.erp_encoder import ERP_encoder_Res
from module.module_elite360m.ico_encoder import ICO_encoder
from module.module_elite360m.fusion import EI_Adaptive_Fusion
from module.module_elite360m.decoder import depth_head, segmentation_head, normal_head, shared_decoder_backbone
from timm.models.layers import trunc_normal_
from module.module_elite360m.decoder_util import Triplet_cross_attention, UpSampleBN_task

os.environ['TORCH_HOME'] = 'pre_model'


class Elite360_ResNet(nn.Module):
    def __init__(self, task, backbone, ico_nblocks=3, ico_nneighbor=32, ico_level=4, embed_channel=64, decode_dim=512,
                 task_decode_dim=128, resolution=512, min_depth_value=0, max_depth_value=10,
                 segmentation_class=13, cross_task=True, training_=True):
        super(Elite360_ResNet, self).__init__()

        self.task = task
        self.task_num = len(self.task.split("_"))
        self.log_var_list = nn.Parameter(torch.zeros((self.task_num,), requires_grad=True))

        assert self.task_num == 3
        self.training_ = training_
        self.cross_task = cross_task
        self.erp_resolution = resolution
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor

        self.erp_backbone = backbone
        self.embed_channel = embed_channel
        self.decode_dim = decode_dim
        self.task_decode_dim = task_decode_dim
        self.min_depth_value = min_depth_value
        self.max_depth_value = max_depth_value
        self.segmentation_class = segmentation_class

        ## Encoder
        self.ERP_encoder = ERP_encoder_Res(channel=self.embed_channel, backbone=self.erp_backbone)
        self.ICO_encoder = ICO_encoder(channel=self.embed_channel, level=self.ico_level, ico_nblocks=self.ico_nblocks,
                                       ico_nneighbor=self.ico_nneighbor)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.erp_resolution, embedding_dim=self.embed_channel)

        self.shared_decoder = shared_decoder_backbone(channel=self.embed_channel, backbone=self.erp_backbone,
                                                      output_dim=self.task_decode_dim, features=self.decode_dim)

        self.preliminary_depth_decoder = nn.Sequential(
            nn.Conv2d(self.task_decode_dim, self.task_decode_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.task_decode_dim),
            nn.LeakyReLU(inplace=True))

        if self.training_:
            self.intermediate_depth_head = nn.Conv2d(self.task_decode_dim, 1, kernel_size=1, stride=1, padding=0)

        self.preliminary_normal_decoder = nn.Sequential(
            nn.Conv2d(self.task_decode_dim, self.task_decode_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.task_decode_dim),
            nn.LeakyReLU(inplace=True))
        if self.training_:
            self.intermediate_normal_head = nn.Conv2d(self.task_decode_dim, 3, kernel_size=1, stride=1, padding=0)

        self.preliminary_seg_decoder = nn.Sequential(
            nn.Conv2d(self.task_decode_dim, self.task_decode_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.task_decode_dim),
            nn.LeakyReLU(inplace=True))
        if self.training_:
            self.intermediate_seg_head = nn.Conv2d(self.task_decode_dim, self.segmentation_class, kernel_size=1,
                                                   stride=1, padding=0)

        if self.cross_task:
            self.multi_task_attention_layers = Triplet_cross_attention(input_dim=self.task_decode_dim * 2, head=4)

        ## Decoder

        self.depth_decoder = nn.ModuleList()
        self.normal_decoder = nn.ModuleList()
        self.seg_decoder = nn.ModuleList()

        if self.erp_backbone == 'res18' or self.erp_backbone == 'res34':
            self.depth_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=128,
                                output_features=self.task_decode_dim))
            self.depth_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=64,
                                output_features=self.embed_channel))
            self.normal_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=128,
                                output_features=self.task_decode_dim))
            self.normal_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=64,
                                output_features=self.embed_channel))
            self.seg_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=128,
                                output_features=self.task_decode_dim))
            self.seg_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=64,
                                output_features=self.embed_channel))
        else:
            self.depth_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=512,
                                output_features=self.task_decode_dim))
            self.depth_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=256,
                                output_features=self.embed_channel))
            self.normal_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=512,
                                output_features=self.task_decode_dim))
            self.normal_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=256,
                                output_features=self.embed_channel))
            self.seg_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim * 2, unified_input=512,
                                output_features=self.task_decode_dim))
            self.seg_decoder.append(
                UpSampleBN_task(task_input=self.task_decode_dim, unified_input=256,
                                output_features=self.embed_channel))
        if "depth" in self.task:
            self.depth_head = depth_head(features=self.embed_channel, scale=4)
        if "normal" in self.task:
            self.normal_head = normal_head(features=self.embed_channel, scale=4)
        if "segmentation" in self.task:
            self.segmentation_head = segmentation_head(features=self.embed_channel, scale=4,
                                                       num_classes=self.segmentation_class)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        erp_bottle_feature, erp_down_sample_list = self.ERP_encoder(erp_rgb)
        outputs = {}
        # _, N_ico, Len = ico_rgb.shape
        ico_feature_set, ico_coordinate = self.ICO_encoder(ico_rgb, ico_coord)

        fusion_feature_ = self.EI_Fusion(erp_bottle_feature, ico_feature_set, ico_coordinate)

        shared_representation = self.shared_decoder(fusion_feature_, erp_down_sample_list[-1])

        preliminary_depth_feature = self.preliminary_depth_decoder(shared_representation)

        if self.training_:
            outputs["inter_pred_depth"] = self.max_depth_value * self.sigmoid(
                self.intermediate_depth_head(preliminary_depth_feature))

        preliminary_normal_feature = self.preliminary_normal_decoder(shared_representation)

        if self.training_:
            outputs["inter_pred_normal"] = self.intermediate_normal_head(preliminary_normal_feature)

        preliminary_seg_feature = self.preliminary_seg_decoder(shared_representation)
        if self.training_:
            outputs["inter_pred_seg"] = self.intermediate_seg_head(preliminary_seg_feature)

        depth_feature = torch.cat([shared_representation, preliminary_depth_feature], dim=1)
        normal_feature = torch.cat([shared_representation, preliminary_normal_feature], dim=1)
        seg_feature = torch.cat([shared_representation, preliminary_seg_feature], dim=1)

        if self.cross_task:
            depth_feature, normal_feature, seg_feature = self.multi_task_attention_layers(depth_feature, normal_feature,
                                                                                          seg_feature)

        depth_feature1 = self.depth_decoder[0](depth_feature, erp_down_sample_list[-2])
        depth_feature2 = self.depth_decoder[1](depth_feature1, erp_down_sample_list[-3])

        normal_feature1 = self.normal_decoder[0](normal_feature, erp_down_sample_list[-2])
        normal_feature2 = self.normal_decoder[1](normal_feature1, erp_down_sample_list[-3])

        seg_feature1 = self.seg_decoder[0](seg_feature, erp_down_sample_list[-2])
        seg_feature2 = self.seg_decoder[1](seg_feature1, erp_down_sample_list[-3])

        outputs["pred_depth"] = self.max_depth_value * self.sigmoid(self.depth_head(depth_feature2))
        outputs["pred_normal"] = self.normal_head(normal_feature2)
        outputs["pred_segmentation"] = self.segmentation_head(seg_feature2)
        return outputs
