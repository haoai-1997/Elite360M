from __future__ import print_function
import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from utils.projection_transformation import get_icosahedron, erp2sphere

VALID_CLASS_IDS_25 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19, 22, 24, 25, 32, 34, 35, 38, 39, 40)


def read_list(list_file):
    multiple_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            multiple_list.append(line.strip().split(" "))
    return multiple_list


class Structured3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024,ico_level=4, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.multiple_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.ico_level = ico_level
        self.vertices, self.faces = get_icosahedron(self.ico_level)
        self.face_set = self.vertices[self.faces]

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        _img = self._load_img(idx)
        if _img.shape[0] != self.h and _img.shape[1] != self.w:
            _img = cv2.resize(_img, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        _depth = self._load_depth(idx)
        if _depth.shape[:2] != _img.shape[:2]:
            print('RESHAPE DEPTH')
            _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        _semseg = self._load_semseg(idx)
        if _semseg.shape[:2] != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        _normals, _normals_mask = self._load_normals(idx)
        if _normals.shape[:2] != _img.shape[:2]:
            _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            _normals_mask = cv2.resize(_normals_mask, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            _img = np.roll(_img, roll_idx, 1)
            _semseg = np.roll(_semseg, roll_idx, 1)
            _normals = np.roll(_normals, roll_idx, 1)
            _normals_mask = np.roll(_normals_mask, roll_idx, 1)
            _depth = np.roll(_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            _img = cv2.flip(_img, 1)
            _depth = cv2.flip(_depth, 1)
            _semseg = cv2.flip(_semseg, 1)
            _normals = cv2.flip(_normals, 1)
            _normals_mask = np.roll(_normals_mask, roll_idx, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(_img)))
        else:
            aug_rgb = _img

        rgb = self.to_tensor(_img.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(_depth, axis=0))
        inputs["gt_semantic"] = torch.from_numpy(_semseg)
        inputs["gt_normal"] = torch.from_numpy(_normals).permute(2, 0, 1)
        inputs["norm_valid_mask"] = torch.from_numpy(_normals_mask).permute(2, 0, 1)

        inputs["val_depth_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                    & ~torch.isnan(inputs["gt_depth"]))
        inputs["val_seg_mask"] = (inputs["gt_semantic"] >= 0)
        N, _, _ = self.face_set.shape
        # N1, _, _ = self.face_set1.shape
        ico_normalized_img = erp2sphere(inputs["normalized_rgb"].permute(1, 2, 0).numpy(),
                                        np.reshape(self.face_set, [-1, 3]))
        ico_normalized_img = np.reshape(ico_normalized_img, [N, -1, 3])
        ico_normalized_img = np.mean(ico_normalized_img, axis=1)

        ico_img = erp2sphere(inputs["rgb"].permute(1, 2, 0).numpy(), np.reshape(self.face_set, [-1, 3]))
        ico_img = np.reshape(ico_img, [N, -1, 3])
        ico_img = np.mean(ico_img, axis=1)

        ico_coord = np.mean(self.face_set, axis=1)

        inputs["ico_img"] = torch.from_numpy(ico_img)
        inputs["ico_normalized_img"] = torch.from_numpy(ico_normalized_img)
        inputs["ico_coord"] = torch.from_numpy(ico_coord)

        return inputs

    def __len__(self):
        return len(self.multiple_list)

    def _load_img(self, index):
        rgb_name = os.path.join(self.root_dir, self.multiple_list[index][0])
        _img = Image.open(rgb_name).convert('RGB')
        _img = np.array(_img, copy=False)
        return _img

    def _load_semseg(self, index):
        semantic_name = os.path.join(self.root_dir, self.multiple_list[index][2])
        _semseg = Image.open(semantic_name)
        _semseg = np.array(_semseg).astype('int32')
        gt_semantic = np.ones_like(_semseg, dtype=np.int32) * -1
        for idx_, value in enumerate(VALID_CLASS_IDS_25):
            gt_semantic[_semseg == value] = idx_

        return gt_semantic

    def _load_depth(self, index):
        depth_name = os.path.join(self.root_dir, self.multiple_list[index][1])
        _depth = cv2.imread(depth_name, -1)
        abs_gt_depth = _depth.astype(np.float32)
        if abs_gt_depth.max() == np.NAN or abs_gt_depth.max() == 0:
            print(self.multiple_list[index][0])
        _depth = abs_gt_depth / abs_gt_depth.max() * self.max_depth_meters
        return _depth

    def _load_normals(self, index):
        normal_name = os.path.join(self.root_dir, self.multiple_list[index][3])
        _normals = Image.open(normal_name)
        _normals = np.array(_normals, dtype=np.float32, copy=False)
        norm_valid_mask = np.logical_not(
            np.logical_and(
                np.logical_and(
                    _normals[:, :, 0] == 0.0, _normals[:, :, 1] == 0.0),
                _normals[:, :, 2] == 0.0))
        norm_valid_mask = norm_valid_mask[:, :, np.newaxis]
        _normals = 2 * _normals / 255. - 1
        return _normals, norm_valid_mask
