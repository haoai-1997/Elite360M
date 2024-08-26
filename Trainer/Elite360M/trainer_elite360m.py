from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import time
import json
import tqdm
import copy
import torch
import torch.optim as optimizer
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True
from tensorboardX import SummaryWriter
import torch.distributed as dist

torch.manual_seed(100)
torch.cuda.manual_seed(100)
import matplotlib.pyplot as plot
from data_loader.stanford2d3d import Stanford2D3D
from data_loader.structured3d import Structured3D
from data_loader.matterport3d import Matterport3D
from network.Elite360_Res import Elite360_ResNet
from utils.visualize import show_flops_params
from losses.Depthloss import BerhuLoss
from losses.Semanticloss import MixSoftmaxCrossEntropyLoss
from losses.Normalloss import Normal_loss
from losses.MultiTaskLoss import MultiTaskLoss_uncertainty
from metric.Depth_metric import Depth_evaluator, compute_depth_metrics
from metric.Semantic_metric import Semantic_evaluator, compute_semantic_metrics
from metric.Normal_metric import Normal_evaluator, compute_normal_metrics

def get_stanford_class_colors():
    return np.load('dataset_tool/stanford2d3d/colors.npy')


def get_structured_class_colors():
    return np.load('dataset_tool/structured3d/structured_colors.npy')


def get_matterport3d_class_colors():
    return np.load('dataset_tool/matterport3d/colors.npy')


class Trainer_:
    def __init__(self, settings):
        self.settings = settings
        self.settings.log_dir = os.path.join(self.settings.log_dir, self.settings.task)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        self.task = self.settings.task
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
            torch.cuda.set_device(self.settings.local_rank)

        if len(self.settings.gpu_devices) > 1:
            dist.init_process_group('nccl', init_method='env://',
                                    world_size=len(self.settings.gpu_devices),
                                    rank=self.settings.local_rank)

        self.log_path = os.path.join(self.settings.log_dir, self.settings.backbone)
        if len(self.settings.gpu_devices) > 1:
            if dist.get_rank() == 0 and not os.path.exists(self.log_path):
                os.makedirs(self.log_path, exist_ok=True)
        else:
            os.makedirs(self.log_path, exist_ok=True)
        if self.settings.dataset == 'S2D3D':
            self.segmentation_class = 13
            train_dataset = Stanford2D3D(self.settings.dataset_rootdir,
                                         'split/split_s2d3d/stanford2d3d_mul_train.txt',
                                         disable_color_augmentation=self.settings.disable_color_augmentation,
                                         disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                         disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                         is_training=True)
            val_dataset = Stanford2D3D(self.settings.dataset_rootdir,
                                       'split/split_s2d3d/stanford2d3d_mul_test.txt',
                                       disable_color_augmentation=self.settings.disable_color_augmentation,
                                       disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                       disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                       is_training=False)
            self.color = get_stanford_class_colors()
        elif self.settings.dataset == 'Struc3D':
            self.segmentation_class = 25
            train_dataset = Structured3D(self.settings.dataset_rootdir,
                                         'split/split_struc3d/structured3d_mul_train.txt',
                                         disable_color_augmentation=self.settings.disable_color_augmentation,
                                         disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                         disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                         is_training=True)
            val_dataset = Structured3D(self.settings.dataset_rootdir,
                                       'split/split_struc3d/structured3d_mul_test.txt',
                                       disable_color_augmentation=self.settings.disable_color_augmentation,
                                       disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                       disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                       is_training=False)
            self.color = get_structured_class_colors()
        elif self.settings.dataset == 'MP3D':
            self.segmentation_class = 41
            train_dataset = Matterport3D(self.settings.dataset_rootdir,
                                         'split/split_matterport3d/matterport3d_mul_train.txt',
                                         disable_color_augmentation=self.settings.disable_color_augmentation,
                                         disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                         disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                         is_training=True)
            val_dataset = Matterport3D(self.settings.dataset_rootdir,
                                       'split/split_matterport3d/matterport3d_mul_test.txt',
                                       disable_color_augmentation=self.settings.disable_color_augmentation,
                                       disable_LR_filp_augmentation=self.settings.disable_LR_filp_augmentation,
                                       disable_yaw_rotation_augmentation=self.settings.disable_yaw_rotation_augmentation,
                                       is_training=False)
            self.color = get_matterport3d_class_colors()
        else:
            raise RuntimeError("No this dataset!!!!")

        self.train_sampler = None if len(
            self.settings.gpu_devices) < 2 else torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=len(self.settings.gpu_devices), rank=self.settings.local_rank)

        self.train_loader = DataLoader(train_dataset, batch_size=self.settings.batch_size,
                                       shuffle=(self.train_sampler is None), num_workers=self.settings.num_workers,
                                       pin_memory=True, sampler=self.train_sampler, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        self.val_sampler = None if len(
            self.settings.gpu_devices) < 2 else torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=len(self.settings.gpu_devices), rank=self.settings.local_rank)

        self.val_loader = DataLoader(val_dataset, batch_size=self.settings.batch_size,
                                     shuffle=(self.val_sampler is None), num_workers=self.settings.num_workers,
                                     pin_memory=True, sampler=self.val_sampler, drop_last=True)

        if self.settings.backbone == "Elite360M_Res34":
            self.model = Elite360_ResNet(task=self.task, backbone='res34', embed_channel=64, ico_nblocks=3,
                                         decode_dim=512, task_decode_dim=128, segmentation_class=self.segmentation_class)
        elif self.settings.backbone == "Elite360M_Res18":
            self.model = Elite360_ResNet(task=self.task, backbone='res18', embed_channel=64, ico_nblocks=3,
                                         decode_dim=512, task_decode_dim=128,segmentation_class=self.segmentation_class)
        elif self.settings.backbone == "Elite360M_Res50":
            self.model = Elite360_ResNet(task=self.task, backbone='res50', embed_channel=128, ico_nblocks=3,
                                         decode_dim=512, task_decode_dim=128,segmentation_class=self.segmentation_class)
        else:
            self.model = None

        if len(self.settings.gpu_devices) > 1:
            process_group = torch.distributed.new_group(list(range(len(self.settings.gpu_devices))))
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)

        self.model.to(self.device)

        if len(self.settings.gpu_devices) > 1:
            if self.settings.local_rank == 0:
                try:
                    show_flops_params(copy.deepcopy(self.model), self.device)
                except Exception as e:
                    print('get flops and params error: {}'.format(e))
        else:
            try:
                show_flops_params(copy.deepcopy(self.model), self.device)
            except Exception as e:
                print('get flops and params error: {}'.format(e))

        if len(self.settings.gpu_devices) > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self.settings.local_rank],
                                                                   output_device=self.settings.local_rank,
                                                                   find_unused_parameters=False)

        self.parameters_to_train = list(self.model.parameters())
        self.optimizer = optimizer.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()
        if len(self.settings.gpu_devices) > 1:
            if dist.get_rank() == 0:
                print("Training models named:\n ", self.settings.backbone)
                print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
                print("Training is using:\n ", self.device)
        else:
            print("Training models named:\n ", self.settings.backbone)
            print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
            print("Training is using:\n ", self.device)

        self.compute_depth_loss = BerhuLoss()
        self.compute_normal_loss = Normal_loss("AL")
        self.compute_seg_loss = MixSoftmaxCrossEntropyLoss()

        self.multi_task_loss = MultiTaskLoss_uncertainty(loss_weights={"depth": 1, "normal": 2, "segmentation": 2})

        self.depth_evaluator = Depth_evaluator()
        self.normal_evaluator = Normal_evaluator()
        self.semantic_evaluator = Semantic_evaluator(self.segmentation_class)

        if self.settings.local_rank == 0:
            self.save_settings()

        self.best_depth_rmse = 20
        self.best_semantic_mIoU = 0
        self.best_normal_mean = 50

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate(self.epoch)
        self.save_last_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)

            inter_loss = losses["inter_" + self.task.split("_")[0] + "_loss"] + \
                         losses["inter_" + self.task.split("_")[1] + "_loss"] + \
                         losses["inter_" + self.task.split("_")[2] + "_loss"]

            if batch_idx % 100 == 0:
                if len(self.settings.gpu_devices) > 1 and self.settings.local_rank == 0:
                    for i in range(len(self.task.split("_"))):
                        print("inter_" + self.task.split("_")[i] + "_loss" + ":" + str(
                            losses["inter_" + self.task.split("_")[i] + "_loss"].item()))
                    print("final_loss:" + str(losses["final_loss"].item()))
            total_loss = inter_loss + losses["final_loss"]
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        losses = {}

        equi_inputs = inputs["normalized_rgb"]
        ico_images = inputs["ico_normalized_img"]
        ico_coords = inputs["ico_coord"]
        depth_mask = inputs["val_depth_mask"]
        normal_mask = inputs["norm_valid_mask"]
        gt_semantic = inputs["gt_semantic"]

        outputs = self.model(equi_inputs, ico_images, ico_coords)

        inter_depth_mask = torch.nn.functional.interpolate(depth_mask.float(), (
            outputs["inter_pred_depth"].size(-2), outputs["inter_pred_depth"].size(-1)), mode='nearest')
        inter_pred_depth = outputs["inter_pred_depth"] * inter_depth_mask
        inter_gt_depth = torch.nn.functional.interpolate(inputs["gt_depth"], (
            outputs["inter_pred_depth"].size(-2), outputs["inter_pred_depth"].size(-1)),
                                                         mode='nearest') * inter_depth_mask
        losses["inter_depth_loss"] = self.compute_depth_loss(inter_gt_depth, inter_pred_depth)

        inter_normal_mask = torch.nn.functional.interpolate(normal_mask.float(), (
            outputs["inter_pred_normal"].size(-2), outputs["inter_pred_normal"].size(-1)), mode='nearest')
        inter_pred_normal = outputs["inter_pred_normal"]
        inter_gt_normal = torch.nn.functional.interpolate(inputs["gt_normal"], (
            outputs["inter_pred_normal"].size(-2), outputs["inter_pred_normal"].size(-1)),
                                                          mode='nearest')
        losses["inter_normal_loss"] = self.compute_normal_loss(inter_gt_normal, inter_pred_normal,
                                                               inter_normal_mask > 0.1)

        inter_gt_seg = torch.nn.functional.interpolate(gt_semantic.float()[:, None, :, :], (
            outputs["inter_pred_seg"].size(-2), outputs["inter_pred_seg"].size(-1)), mode='nearest')[:, 0, :, :]
        losses["inter_segmentation_loss"] = self.compute_seg_loss(outputs["inter_pred_seg"], inter_gt_seg)

        outputs["pred_depth"] = outputs["pred_depth"] * depth_mask
        outputs["pred_normal"] = outputs["pred_normal"] * normal_mask

        losses["final_loss"] = self.multi_task_loss(inputs["gt_depth"][inputs["val_depth_mask"]],
                                                    outputs["pred_depth"][inputs["val_depth_mask"]],
                                                    outputs["pred_normal"], inputs["gt_normal"],
                                                    normal_mask, outputs["pred_segmentation"], gt_semantic.float(),
                                                    self.model.module.log_var_list)
        return outputs, losses

    def validate(self, epoch):
        """Validate the models on the validation set
        """
        self.model.eval()
        if "depth" in self.task:
            self.depth_evaluator.reset_eval_metrics()

        if "segmentation" in self.task:
            self.semantic_evaluator.reset_eval_metrics()

        if "normal" in self.task:
            self.normal_evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))
        evaluation_errors = {}
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs)

                depth_mask = inputs["val_depth_mask"]

                normal_mask = inputs["norm_valid_mask"]
                if "depth" in self.task:
                    pred_depth = outputs["pred_depth"].detach() * depth_mask
                    gt_depth = inputs["gt_depth"].detach() * depth_mask
                    self.depth_evaluator.compute_eval_metrics(gt_depth, pred_depth, depth_mask)
                if "segmentation" in self.task:
                    pred_semantic = outputs["pred_segmentation"].detach()
                    gt_semantic = inputs["gt_semantic"].detach()
                    self.semantic_evaluator.compute_eval_metrics(pred_semantic, gt_semantic)
                if "normal" in self.task:
                    pred_normal = outputs["pred_normal"].detach() * normal_mask
                    gt_normal = inputs["gt_normal"].detach() * normal_mask
                    self.normal_evaluator.compute_eval_metrics(gt_normal, pred_normal, normal_mask)

                if self.settings.local_rank == 0 or len(self.settings.gpu_devices) == 1:
                    rgb_img = inputs["rgb"].detach().cpu().numpy()

                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                    if "depth" in self.task:
                        if batch_idx % 100 == 0 and epoch % 1 == 0:
                            depth_prediction = pred_depth.detach().cpu().numpy()
                            gt_prediction = gt_depth.detach().cpu().numpy()
                            depth_vis_dir = os.path.join(self.log_path, "visual_results", "depth",
                                                         str(epoch) + "_" + str(batch_idx))
                            if not os.path.exists(depth_vis_dir):
                                os.makedirs(depth_vis_dir)
                            #### depth visualization
                            cv2.imwrite('{}/test_equi_rgb.png'.format(depth_vis_dir),
                                        rgb_img[:, :, ::-1] * 255)
                            plot.imsave('{}/test_equi_depth_pred.png'.format(depth_vis_dir),
                                        depth_prediction[0, 0, :, :], cmap="jet")
                            plot.imsave('{}/test_equi_depth_gt.png'.format(depth_vis_dir),
                                        gt_prediction[0, 0, :, :], cmap="jet")
                    if "segmentation" in self.task:
                        if batch_idx % 100 == 0 and epoch % 1 == 0:
                            semantic_vis_dir = os.path.join(self.log_path, "visual_results", "semantic",
                                                            str(epoch) + "_" + str(batch_idx))
                            if not os.path.exists(semantic_vis_dir):
                                os.makedirs(semantic_vis_dir)
                            semantic_prediction = pred_semantic.detach().cpu().numpy()
                            gt_semantic = gt_semantic.detach().cpu().numpy()
                            #### semantic visualization
                            pred_label = np.argmax(semantic_prediction, 1)[0]
                            label = gt_semantic[0]
                            result_semantic = np.zeros(
                                (pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)
                            class_colors = self.color
                            label_semantic = np.zeros(
                                (label.shape[0], label.shape[1], 3), dtype=np.uint8)
                            mask_semantic = np.zeros(
                                (label.shape[0], label.shape[1], 3), dtype=np.uint8)
                            for x in range(pred_label.shape[0]):
                                for y in range(pred_label.shape[1]):
                                    if pred_label[x][y] >= 0:
                                        result_semantic[x,
                                                        y] = class_colors[pred_label[x][y]]
                                    if label[x][y] >= 0:
                                        mask_semantic[x, y] = 1
                                        label_semantic[x, y] = class_colors[label[x][y]]
                            vis_image = result_semantic // 2 + rgb_img[:, :, ::-1] * 255 // 2
                            vis_image = vis_image * mask_semantic
                            result_semantic = mask_semantic * result_semantic
                            cv2.imwrite('{}/test_equi_rgb.png'.format(semantic_vis_dir),
                                        rgb_img[:, :, ::-1] * 255)
                            cv2.imwrite('{}/test_equi_vis.png'.format(semantic_vis_dir),
                                        vis_image)
                            cv2.imwrite('{}/test_equi_semantic_pred.png'.format(semantic_vis_dir),
                                        result_semantic)
                            cv2.imwrite('{}/test_equi_semantic_gt.png'.format(semantic_vis_dir),
                                        label_semantic)
                    if "normal" in self.task:
                        if batch_idx % 100 == 0 and epoch % 1 == 0:
                            normal_vis_dir = os.path.join(self.log_path, "visual_results", "normal",
                                                          str(epoch) + "_" + str(batch_idx))
                            if not os.path.exists(normal_vis_dir):
                                os.makedirs(normal_vis_dir)
                            pred_norm = pred_normal[0:1, :3, :, :].detach().cpu().numpy()
                            pred_norm = pred_norm.transpose(0, 2, 3, 1)  # (B, H, W, 3)
                            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
                            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
                            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
                            plot.imsave('{}/test_equi_normal_pred.png'.format(normal_vis_dir),
                                        pred_norm_rgb[0, :, :, :])

                            gt_norm = gt_normal[0:1, :3, :, :].detach().cpu().numpy()
                            gt_norm = gt_norm.transpose(0, 2, 3, 1)  # (B, H, W, 3)
                            gt_norm_rgb = ((gt_norm + 1) * 0.5) * 255
                            gt_norm_rgb = np.clip(gt_norm_rgb, a_min=0, a_max=255)
                            gt_norm_rgb = gt_norm_rgb.astype(np.uint8)  # (B, H, W, 3)

                            plot.imsave('{}/test_equi_normal_gt.png'.format(normal_vis_dir),
                                        gt_norm_rgb[0, :, :, :])
                            cv2.imwrite('{}/test_equi_rgb.png'.format(normal_vis_dir),
                                        rgb_img[:, :, ::-1] * 255)

            if len(self.settings.gpu_devices) > 1 and self.settings.local_rank == 0:
                if "depth" in self.task:
                    if not os.path.exists(os.path.join(self.log_path, "depth")):
                        os.makedirs(os.path.join(self.log_path, "depth"))
                    self.depth_evaluator.print(dir=os.path.join(self.log_path, "depth"))
                    for i, key in enumerate(self.depth_evaluator.metrics.keys()):
                        evaluation_errors["depth/" + key] = np.array(self.depth_evaluator.metrics[key].avg.cpu())
                    self.current_depth_rmse = np.array(self.depth_evaluator.metrics["err/rms"].avg.cpu()).item()
                    if self.current_depth_rmse < self.best_depth_rmse:
                        self.best_depth_rmse = self.current_depth_rmse
                        if (self.epoch + 1) % self.settings.save_frequency == 0 and self.settings.local_rank == 0:
                            self.save_best_depth_model()
                if "segmentation" in self.task:
                    if not os.path.exists(os.path.join(self.log_path, "semantic")):
                        os.makedirs(os.path.join(self.log_path, "semantic"))
                    self.semantic_evaluator.print(dir=os.path.join(self.log_path, "semantic"))
                    for i, key in enumerate(self.semantic_evaluator.metrics.keys()):
                        evaluation_errors["semantic/" + key] = np.array(
                            self.semantic_evaluator.metrics[key].avg.cpu())
                    self.current_semantic_mIoU = np.array(
                        self.semantic_evaluator.metrics["err/mIoU"].avg.cpu()).item()
                    if self.current_semantic_mIoU > self.best_semantic_mIoU:
                        self.best_semantic_mIoU = self.current_semantic_mIoU
                        if (self.epoch + 1) % self.settings.save_frequency == 0 and self.settings.local_rank == 0:
                            self.save_best_semantic_model()
                if "normal" in self.task:
                    if not os.path.exists(os.path.join(self.log_path, "normal")):
                        os.makedirs(os.path.join(self.log_path, "normal"))
                    self.normal_evaluator.print(dir=os.path.join(self.log_path, "normal"))
                    for i, key in enumerate(self.normal_evaluator.metrics.keys()):
                        evaluation_errors["normal/" + key] = np.array(self.normal_evaluator.metrics[key].avg.cpu())
                    self.current_normal_mean = np.array(self.normal_evaluator.metrics["err/mean"].avg.cpu()).item()
                    if self.current_normal_mean < self.best_normal_mean:
                        self.best_normal_mean = self.current_normal_mean
                        if (self.epoch + 1) % self.settings.save_frequency == 0 and self.settings.local_rank == 0:
                            self.save_best_normal_model()
        del inputs, outputs, losses, evaluation_errors

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_best_depth_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "depth_models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("best_depth_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

    def save_best_semantic_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "semantic_models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("best_semantic_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

    def save_best_normal_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "normal_models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("best_normal_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

    def save_last_model(self):
        """Save models weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format("last"))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        if self.settings.local_rank == 0:
            save_path = os.path.join(save_folder, "{}.pth".format("last_model"))
            to_save = self.model.module.state_dict()
            torch.save(to_save, save_path)

    def load_model(self):
        """Load models from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        if self.settings.local_rank == 0:
            print("loading models from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("best_model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
