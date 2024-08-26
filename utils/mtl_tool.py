from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np

class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()

def create_task_flags(task, seg_class,):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    total_tasks = {'segmentation': seg_class, 'depth': 1, 'normal': 3}
    tasks = {}
    if task !="all":
        for i in range(len(task.split("_"))):
            tasks[task.split("_")[i]] = total_tasks[task.split("_")[i]]
    else:
        tasks = total_tasks
    return tasks

class TaskMetric:
    def __init__(self, pri_tasks, batch_size, epochs):
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.metric = {key: np.zeros([epochs]) for key in pri_tasks.keys()}
        self.data_counter = 0
        self.epoch_counter = 0

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

    def update_metric(self, task_pred, task_loss):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_loss: {''TASK_ID1': TASK_LOSS1, 'TASK_ID2': TASK_LOSS2, ...}
        """
        curr_bs = task_pred[0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1
        with torch.no_grad():
            for task_id, loss in task_loss.items():
                self.metric[task_id][e] = r * self.metric[task_id][e] + (1 - r) * loss.item()
