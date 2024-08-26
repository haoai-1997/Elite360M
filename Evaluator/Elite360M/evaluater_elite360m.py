from __future__ import absolute_import, division, print_function
import os

import numpy as np

import tqdm
import copy
import torch

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

torch.manual_seed(100)
torch.cuda.manual_seed(100)
import matplotlib.pyplot as plot
from data_loader.stanford2d3d import Stanford2D3D
from data_loader.structured3d import Structured3D
from data_loader.matterport3d import Matterport3D
from network.Elite360_Res import Elite360_ResNet
from utils.visualize import show_flops_params
from metric.Depth_metric import Depth_evaluator, compute_depth_metrics
from metric.Semantic_metric import Semantic_evaluator, compute_semantic_metrics
from metric.Normal_metric import Normal_evaluator, compute_normal_metrics
import cv2


def get_stanford_class_colors():
    return np.load('dataset_tool/stanford2d3d/colors.npy')


def get_structured_class_colors():
    return np.load('dataset_tool/structured3d/structured_colors.npy')


def get_matterport3d_class_colors():
    return np.load('dataset_tool/matterport3d/colors.npy')


class Evaluator_:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        self.task = self.settings.task

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir,
                                     self.settings.backbone, self.settings.task)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)

        if self.settings.dataset == 'S2D3D':
            self.segmentation_class = 13
            val_dataset = Stanford2D3D(self.settings.dataset_rootdir,
                                       'split/split_s2d3d/stanford2d3d_mul_test.txt',
                                       disable_color_augmentation=True,
                                       disable_LR_filp_augmentation=True,
                                       disable_yaw_rotation_augmentation=True,
                                       is_training=False)
            self.color = get_stanford_class_colors()
        elif self.settings.dataset == 'Struc3D':
            self.segmentation_class = 25
            val_dataset = Structured3D(self.settings.dataset_rootdir,
                                       'split/split_struc3d/structured3d_mul_test.txt',
                                       disable_color_augmentation=True,
                                       disable_LR_filp_augmentation=True,
                                       disable_yaw_rotation_augmentation=True,
                                       is_training=False)
            self.color = get_structured_class_colors()
        elif self.settings.dataset == 'MP3D':
            self.segmentation_class = 41
            val_dataset = Matterport3D(self.settings.dataset_rootdir,
                                       'split/split_matterport3d/matterport3d_mul_test.txt',
                                       disable_color_augmentation=True,
                                       disable_LR_filp_augmentation=True,
                                       disable_yaw_rotation_augmentation=True,
                                       is_training=False)
            self.color = get_matterport3d_class_colors()
        else:
            raise RuntimeError("No this dataset!!!!")

        self.val_loader = DataLoader(val_dataset, batch_size=self.settings.batch_size,
                                     shuffle=False, num_workers=self.settings.num_workers,
                                     pin_memory=True, drop_last=False)

        if self.settings.backbone == "Elite360M_Res18":
            self.model_depth = Elite360_ResNet(task=self.task, backbone='res18', embed_channel=64, ico_nblocks=3,
                                               decode_dim=512, task_decode_dim=128,
                                               segmentation_class=self.segmentation_class)
            self.model_normal = Elite360_ResNet(task=self.task, backbone='res18', embed_channel=64, ico_nblocks=3,
                                                decode_dim=512, task_decode_dim=128,
                                                segmentation_class=self.segmentation_class)
            self.model_segmentation = Elite360_ResNet(task=self.task, backbone='res18', embed_channel=64, ico_nblocks=3,
                                                      decode_dim=512, task_decode_dim=128,
                                                      segmentation_class=self.segmentation_class)
        elif self.settings.backbone == "Elite360M_Res34":
            self.model_depth = Elite360_ResNet(task=self.task, backbone='res34', embed_channel=64, ico_nblocks=3,
                                               decode_dim=512, task_decode_dim=128,
                                               segmentation_class=self.segmentation_class)
            self.model_normal = Elite360_ResNet(task=self.task, backbone='res34', embed_channel=64, ico_nblocks=3,
                                                decode_dim=512, task_decode_dim=128,
                                                segmentation_class=self.segmentation_class)
            self.model_segmentation = Elite360_ResNet(task=self.task, backbone='res34', embed_channel=64, ico_nblocks=3,
                                                      decode_dim=512, task_decode_dim=128,
                                                      segmentation_class=self.segmentation_class)
        elif self.settings.backbone == "Elite360M_Res50":
            self.model_depth = Elite360_ResNet(task=self.task, backbone='res50', embed_channel=128, ico_nblocks=3,
                                               decode_dim=512, task_decode_dim=128,
                                               segmentation_class=self.segmentation_class)
            self.model_normal = Elite360_ResNet(task=self.task, backbone='res50', embed_channel=128, ico_nblocks=3,
                                                decode_dim=512, task_decode_dim=128,
                                                segmentation_class=self.segmentation_class)
            self.model_segmentation = Elite360_ResNet(task=self.task, backbone='res50', embed_channel=128,
                                                      ico_nblocks=3,
                                                      decode_dim=512, task_decode_dim=128,
                                                      segmentation_class=self.segmentation_class)
        else:
            self.model_depth = None
            self.model_normal = None
            self.model_segmentation = None

        if self.settings.load_weights_dir is None:
            print("No load Model")
            import sys
            sys.exit()
        self.evaluator = {}

        if "depth" in self.task:
            self.model_depth.to(self.device)
            self.load_depth_model(
                os.path.join(self.settings.load_weights_dir, self.task, self.settings.backbone, "models/weights_last"))
            self.evaluator["depth"] = Depth_evaluator()
            try:
                show_flops_params(copy.deepcopy(self.model_depth), self.device)
            except Exception as e:
                print('get flops and params error: {}'.format(e))
        if "segmentation" in self.task:
            self.model_segmentation.to(self.device)
            self.load_semantic_model(
                os.path.join(self.settings.load_weights_dir, self.task, self.settings.backbone, "models/weights_last"))
            self.evaluator["segmentation"] = Semantic_evaluator(self.segmentation_class)
            try:
                show_flops_params(copy.deepcopy(self.model_segmentation), self.device)
            except Exception as e:
                print('get flops and params error: {}'.format(e))
        if "normal" in self.task:
            self.model_normal.to(self.device)
            self.load_normal_model(
                os.path.join(self.settings.load_weights_dir, self.task, self.settings.backbone, "models/weights_last"))
            self.evaluator["normal"] = Normal_evaluator()
            try:
                show_flops_params(copy.deepcopy(self.model_normal), self.device)
            except Exception as e:
                print('get flops and params error: {}'.format(e))
        print("Metrics are saved to:\n", self.log_path)

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        output_set = {}

        equi_inputs = inputs["normalized_rgb"]
        ico_images = inputs["ico_normalized_img"]
        ico_coords = inputs["ico_coord"]

        if "depth" in self.task:
            outputs = self.model_depth(equi_inputs, ico_images, ico_coords)
            output_set["pred_depth"] = outputs["pred_depth"]
            output_set["inter_pred_depth"] = outputs["inter_pred_depth"]
        if "normal" in self.task:
            outputs = self.model_normal(equi_inputs, ico_images, ico_coords)
            output_set["pred_normal"] = outputs["pred_normal"]
            output_set["inter_pred_normal"] = outputs["inter_pred_normal"]
        if "segmentation" in self.task:
            outputs = self.model_segmentation(equi_inputs, ico_images, ico_coords)
            output_set["pred_segmentation"] = outputs["pred_segmentation"]
            output_set["inter_pred_seg"] = outputs["inter_pred_seg"]
        return output_set

    def validate(self):
        """Validate the models on the validation set
        """
        self.model_depth.eval()
        self.model_normal.eval()
        self.model_segmentation.eval()
        if "depth" in self.task:
            self.evaluator["depth"].reset_eval_metrics()
        if "segmentation" in self.task:
            self.evaluator["segmentation"].reset_eval_metrics()
        if "normal" in self.task:
            self.evaluator["normal"].reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs = self.process_batch(inputs)
                depth_mask = inputs["val_depth_mask"]

                normal_mask = inputs["norm_valid_mask"]
                pred = {}
                gt = {}
                mask = inputs["val_seg_mask"]
                if "depth" in self.task:
                    pred_depth = outputs["pred_depth"].detach() * depth_mask
                    gt_depth = inputs["gt_depth"].detach() * depth_mask
                    self.evaluator["depth"].compute_eval_metrics(gt_depth, pred_depth, depth_mask)
                    pred["depth"] = pred_depth
                    gt["depth"] = gt_depth
                    inter_depth_mask = torch.nn.functional.interpolate(depth_mask.float(), (
                        outputs["inter_pred_depth"].size(-2), outputs["inter_pred_depth"].size(-1)), mode='nearest')
                    pred["inter_pred_depth"] = outputs["inter_pred_depth"].detach() * inter_depth_mask
                if "normal" in self.task:
                    pred_normal = outputs["pred_normal"].detach() * normal_mask
                    gt_normal = inputs["gt_normal"].detach() * normal_mask
                    self.evaluator["normal"].compute_eval_metrics(gt_normal, pred_normal, normal_mask)
                    inter_pred_normal = outputs["inter_pred_normal"].detach()
                    pred["normal"] = pred_normal
                    pred["inter_pred_normal"] = inter_pred_normal
                    gt["normal"] = gt_normal
                if "segmentation" in self.task:
                    pred_semantic = outputs["pred_segmentation"].detach()
                    gt_semantic = inputs["gt_semantic"].detach()
                    self.evaluator["segmentation"].compute_eval_metrics(pred_semantic, gt_semantic)
                    pred["segmentation"] = pred_semantic
                    pred["inter_pred_seg"] = outputs["inter_pred_seg"].detach()
                    gt["segmentation"] = gt_semantic
                if batch_idx % self.settings.log_frequency == 0:
                    rgb_img = inputs["rgb"].detach().cpu().numpy()

                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)

                    if "depth" in self.task:
                        depth_vis_dir = os.path.join(self.log_path, "visual_results", "depth", str(batch_idx))
                        depth_record_dir = os.path.join(self.log_path, "depth")
                        if not os.path.exists(depth_vis_dir):
                            os.makedirs(depth_vis_dir, exist_ok=True)
                        if not os.path.exists(depth_record_dir):
                            os.makedirs(depth_record_dir, exist_ok=True)
                        self.evaluator["depth"].print(dir=depth_record_dir)
                        depth_prediction = pred["depth"].detach().cpu().numpy()
                        gt_prediction = gt["depth"].detach().cpu().numpy()
                        #### depth visualization
                        cv2.imwrite('{}/test_equi_rgb.png'.format(depth_vis_dir),
                                    rgb_img[:, :, ::-1] * 255)
                        plot.imsave('{}/test_equi_depth_pred.png'.format(depth_vis_dir),
                                    depth_prediction[0, 0, :, :], cmap="jet")
                        plot.imsave('{}/test_equi_depth_gt.png'.format(depth_vis_dir),
                                    gt_prediction[0, 0, :, :], cmap="jet")
                    if "normal" in self.task:
                        normal_vis_dir = os.path.join(self.log_path, "visual_results", "normal", str(batch_idx))
                        if not os.path.exists(normal_vis_dir):
                            os.makedirs(normal_vis_dir, exist_ok=True)
                        normal_record_dir = os.path.join(self.log_path, "normal")
                        if not os.path.exists(normal_record_dir):
                            os.makedirs(normal_record_dir, exist_ok=True)
                        self.evaluator["normal"].print(dir=normal_record_dir)

                        normal_prediction = pred["normal"].detach().cpu().numpy()
                        gt_normal = gt["normal"].detach().cpu().numpy()

                        #### normal visualization
                        pred_norm = normal_prediction[:, :3, :, :]
                        pred_norm = pred_norm.transpose(0, 2, 3, 1)  # (B, H, W, 3)
                        pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
                        pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
                        pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
                        plot.imsave('{}/test_equi_normal_pred.png'.format(normal_vis_dir),
                                    pred_norm_rgb[0, :, :, :])

                        gt_norm = gt_normal[:, :3, :, :]
                        gt_norm = gt_norm.transpose(0, 2, 3, 1)  # (B, H, W, 3)
                        gt_norm_rgb = ((gt_norm + 1) * 0.5) * 255
                        gt_norm_rgb = np.clip(gt_norm_rgb, a_min=0, a_max=255)
                        gt_norm_rgb = gt_norm_rgb.astype(np.uint8)  # (B, H, W, 3)

                        plot.imsave('{}/test_equi_normal_gt.png'.format(normal_vis_dir),
                                    gt_norm_rgb[0, :, :, :])
                    if "segmentation" in self.task:
                        semantic_vis_dir = os.path.join(self.log_path, "visual_results", "semantic", str(batch_idx))
                        if not os.path.exists(semantic_vis_dir):
                            os.makedirs(semantic_vis_dir, exist_ok=True)
                        semantic_record_dir = os.path.join(self.log_path, "semantic")
                        if not os.path.exists(semantic_record_dir):
                            os.makedirs(semantic_record_dir, exist_ok=True)

                        self.evaluator["segmentation"].print(dir=semantic_record_dir)

                        semantic_prediction = pred["segmentation"].detach().cpu().numpy()
                        gt_semantic = gt["segmentation"].detach().cpu().numpy()

                        if self.settings.dataset != "Struc3D":
                            pred_label = np.argmax(semantic_prediction, 1) * mask.cpu().numpy() + (
                                    mask.int().cpu().numpy() - 1)
                            pred_label = pred_label[0] + 1
                            label = gt_semantic[0] + 1
                        else:
                            pred_label = np.argmax(semantic_prediction, 1) * mask.cpu().numpy() + (
                                    mask.int().cpu().numpy() - 1)
                            pred_label = pred_label[0]
                            label = gt_semantic[0]
                        result_semantic = np.zeros(
                            (pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)
                        if self.settings.dataset == "MP3D":
                            class_colors = self.color * 255
                        else:
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
                        cv2.imwrite('{}/test_equi_vis.png'.format(semantic_vis_dir),
                                    vis_image)
                        cv2.imwrite('{}/test_equi_semantic_pred.png'.format(semantic_vis_dir),
                                    result_semantic)
                        cv2.imwrite('{}/test_equi_semantic_gt.png'.format(semantic_vis_dir),
                                    label_semantic)
        if "depth" in self.task:
            if os.path.exists(os.path.join(self.log_path, "depth")):
                os.makedirs(os.path.join(self.log_path, "depth"), exist_ok=True)
            self.evaluator["depth"].print(dir=os.path.join(self.log_path, "depth"))
        if "normal" in self.task:
            if os.path.exists(os.path.join(self.log_path, "normal")):
                os.makedirs(os.path.join(self.log_path, "normal"), exist_ok=True)
            self.evaluator["normal"].print(dir=os.path.join(self.log_path, "normal"))
        if "segmentation" in self.task:
            if os.path.exists(os.path.join(self.log_path, "semantic")):
                os.makedirs(os.path.join(self.log_path, "semantic"), exist_ok=True)
            self.evaluator["segmentation"].print(dir=os.path.join(self.log_path, "semantic"))

        del inputs, outputs, pred, gt

    def load_depth_model(self, dir):
        """Load models from disk
        """
        dir = os.path.expanduser(dir)

        assert os.path.isdir(dir), \
            "Cannot find folder {}".format(dir)
        print("loading models from folder {}".format(dir))

        path = os.path.join(dir, "{}.pth".format("last_model"))
        model_dict = self.model_depth.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model_depth.load_state_dict(model_dict, strict=True)

    def load_semantic_model(self, dir):
        """Load models from disk
        """
        dir = os.path.expanduser(dir)

        assert os.path.isdir(dir), \
            "Cannot find folder {}".format(dir)
        print("loading models from folder {}".format(dir))

        path = os.path.join(dir, "{}.pth".format("last_model"))  ## edit
        model_dict = self.model_segmentation.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model_segmentation.load_state_dict(model_dict, strict=True)

    def load_normal_model(self, dir):
        """Load models from disk
        """
        dir = os.path.expanduser(dir)

        assert os.path.isdir(dir), \
            "Cannot find folder {}".format(dir)
        print("loading models from folder {}".format(dir))

        path = os.path.join(dir, "{}.pth".format("last_model"))
        model_dict = self.model_normal.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model_normal.load_state_dict(model_dict, strict=True)
