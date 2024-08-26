import os
import torch


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0)  # .item()
    pixel_correct = torch.sum((predict == target) * (target > 0))  # .item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(
        intersection, bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item(
    ) == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def compute_semantic_metrics(pred, label, n_class):
    correct, labeled = batch_pix_accuracy(pred, label)
    inter, union = batch_intersection_union(pred, label, n_class)
    pixAcc = 1.0 * correct / \
             (2.220446049250313e-16 + labeled)  # remove np.spacing(1)
    IoU = 1.0 * inter / \
          (2.220446049250313e-16 + union)
    mIoU = IoU.mean()
    return pixAcc.mean(), mIoU


# From https://github.com/fyu/drn
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


class Semantic_evaluator(object):

    def __init__(self, n_class):
        self.n_class = n_class
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/pixAcc"] = AverageMeter()
        self.metrics["err/mIoU"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the models
        """
        self.metrics["err/pixAcc"].reset()
        self.metrics["err/mIoU"].reset()

    def compute_eval_metrics(self, gt_semantic, pred_semantic):
        """
        Computes metrics used to evaluate the models
        """
        N = gt_semantic.shape[0]

        pixAcc, mIoU = compute_semantic_metrics(gt_semantic, pred_semantic, self.n_class)

        self.metrics["err/pixAcc"].update(pixAcc, N)
        self.metrics["err/mIoU"].update(mIoU, N)

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/pixAcc"].avg)
        avg_metrics.append(self.metrics["err/mIoU"].avg)

        print("\n  " + ("{:>9} | " * 2).format("pixAcc", "mIoU"))
        print(("&  {: 8.5f} " * 2).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'a+') as f:
                print("\n  " + ("{:>9} | " * 2).format("pixAcc", "mIoU"), file=f)
                print(("&  {: 8.5f} " * 2).format(*avg_metrics), file=f)
