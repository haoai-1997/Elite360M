import os
import torch
import numpy as np


def compute_normal_metrics(gt, pred, mask):
    """Computation of metrics between predicted and ground truth depths
    """
    mask = mask[:, 0, :, :]


    prediction_error = torch.cosine_similarity(pred, gt, dim=1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    E = torch.acos(prediction_error) * 180.0 / np.pi
    normal_errors = E[mask]

    mean = torch.mean(normal_errors)
    median = torch.median(normal_errors)

    rmse = normal_errors ** 2
    rmse = torch.sqrt(rmse.mean())
    a1 = 100.0 * (normal_errors < 5).float().mean()
    a2 = 100.0 * (normal_errors < 7.5).float().mean()
    a3 = 100.0 * (normal_errors < 11.25).float().mean()
    a4 = 100.0 * (normal_errors < 22.5).float().mean()
    a5 = 100.0 * (normal_errors < 30).float().mean()

    return mean, median, rmse, a1, a2, a3, a4, a5


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Normal_evaluator(object):
    def __init__(self):
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/mean"] = AverageMeter()
        self.metrics["err/median"] = AverageMeter()
        self.metrics["err/rmse"] = AverageMeter()
        self.metrics["err/a1"] = AverageMeter()
        self.metrics["err/a2"] = AverageMeter()
        self.metrics["err/a3"] = AverageMeter()
        self.metrics["err/a4"] = AverageMeter()
        self.metrics["err/a5"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the models
        """
        self.metrics["err/mean"].reset()
        self.metrics["err/median"].reset()
        self.metrics["err/rmse"].reset()
        self.metrics["err/a1"].reset()
        self.metrics["err/a2"].reset()
        self.metrics["err/a3"].reset()
        self.metrics["err/a4"].reset()
        self.metrics["err/a5"].reset()

    def compute_eval_metrics(self, gt_norm, pred_norm, gt_norm_mask):
        """
        Computes metrics used to evaluate the models
        """
        N = gt_norm.shape[0]

        mean, median, rmse, a1, a2, a3, a4, a5 = \
            compute_normal_metrics(gt_norm, pred_norm, gt_norm_mask)

        self.metrics["err/mean"].update(mean, N)
        self.metrics["err/median"].update(median, N)
        self.metrics["err/rmse"].update(rmse, N)
        self.metrics["err/a1"].update(a1, N)
        self.metrics["err/a2"].update(a2, N)
        self.metrics["err/a3"].update(a3, N)
        self.metrics["err/a4"].update(a4, N)
        self.metrics["err/a5"].update(a5, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/mean"].avg)
        avg_metrics.append(self.metrics["err/median"].avg)
        avg_metrics.append(self.metrics["err/rmse"].avg)
        avg_metrics.append(self.metrics["err/a1"].avg)
        avg_metrics.append(self.metrics["err/a2"].avg)
        avg_metrics.append(self.metrics["err/a3"].avg)
        avg_metrics.append(self.metrics["err/a4"].avg)
        avg_metrics.append(self.metrics["err/a5"].avg)

        print("\n  " + ("{:>9} | " * 8).format("mean", "median", "rmse", "a1", "a2", "a3", "a4", "a5"))
        print(("&  {: 8.5f} " * 8).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'a+') as f:
                print("\n  " + ("{:>9} | " * 8).format("mean", "median", "rmse", "a1", "a2", "a3", "a4", "a5"), file=f)
                print(("&  {: 8.5f} " * 8).format(*avg_metrics), file=f)
