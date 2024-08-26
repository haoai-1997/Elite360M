import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.Depthloss import BerhuLoss
from losses.Semanticloss import MixSoftmaxCrossEntropyLoss
from losses.Normalloss import Normal_loss


class MultiTaskLoss_uncertainty(nn.Module):
    def __init__(self, loss_weights):
        super(MultiTaskLoss_uncertainty, self).__init__()

        self.loss_weights = loss_weights

        self.compute_depth_loss = BerhuLoss()
        self.compute_normal_loss = Normal_loss("AL")
        self.compute_seg_loss = MixSoftmaxCrossEntropyLoss()

    def forward(self, pred_depth, gt_depth, pred_normal, gt_normal, normal_mask, pred_seg, gt_seg, log_vars):
        loss_weight_puls = {}
        loss_weight_mul = {}
        loss_weight_mul["depth"] = torch.exp(-log_vars[0])
        loss_weight_puls["depth"] = log_vars[0]
        loss_weight_mul["normal"] = torch.exp(-log_vars[1])
        loss_weight_puls["normal"] = log_vars[1]
        loss_weight_mul["seg"] = torch.exp(-log_vars[2])
        loss_weight_puls["seg"] = log_vars[2]
        loss_depth = self.loss_weights["depth"] * loss_weight_mul["depth"] * 10 * self.compute_depth_loss(gt_depth,
                                                                                                          pred_depth) + \
                     loss_weight_puls["depth"]
        loss_normal = self.loss_weights["normal"] * loss_weight_mul["normal"] * 10 * self.compute_normal_loss(
            pred_normal,
            gt_normal, normal_mask) + \
                      loss_weight_puls["normal"]
        loss_seg = self.loss_weights["segmentation"] * loss_weight_mul["seg"] * 10 * self.compute_seg_loss(pred_seg,
                                                                                                  gt_seg) + \
                   loss_weight_puls["seg"]

        total_loss = loss_depth + loss_normal + loss_seg
        return total_loss
