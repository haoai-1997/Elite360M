from __future__ import absolute_import, division, print_function
import os
import argparse
import sys

sys.path.insert(0, r'/')

from Trainer.Elite360M.trainer_elite360m import Trainer_

parser = argparse.ArgumentParser(description="360 Degree Panorama Multi-task Training")

# system settings
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")
parser.add_argument('--local_rank', type=int, default=-1, help="the process to be lanched")
# models settings
parser.add_argument("--backbone", type=str, default="Elite360M_Res18",
                    choices=['Elite360M_Res18','Elite360M_Res34','Elite360M_Res50'],
                    help="backbone")
parser.add_argument("--task", type=str, default="depth",
                    choices=['depth', 'segmentation', 'normal', 'depth_segmentation', 'depth_normal',
                             'normal_segmentation','depth_normal_segmentation'],
                    help="select task to train")
parser.add_argument("--dataset_rootdir", default=None, type=str,help="root dir of dataset")
# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")

# loading and logging settings
parser.add_argument("--load_weights_dir", default=None, type=str,
                    help="folder of models to load")  # , default='./tmp/panodepth/models/weights_pretrain'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "workdirs_test"),
                    help="log directory")
parser.add_argument("--log_frequency", type=int, default=200, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=10, help="number of epochs between each save")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")
# dataset settings
parser.add_argument("--dataset", default='S2D3D', type=str, choices=['S2D3D', 'Struc3D', 'MP3D'],
                    help="Training dataset")

args = parser.parse_args()


def main():
    trainer = Trainer_(args)
    trainer.train()

if __name__ == "__main__":
    main()
