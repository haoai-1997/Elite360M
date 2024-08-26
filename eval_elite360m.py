from __future__ import absolute_import, division, print_function
import os
import sys
import argparse

sys.path.insert(0, r'/')

from Evaluator.Elite360M.evaluater_elite360m import Evaluator_

parser = argparse.ArgumentParser(description="360 Degree Panorama Multi-task Evaluating")
# models settings
parser.add_argument("--backbone", type=str, default="Elite360M_Res18",
                    choices=['Elite360M_Res18','Elite360M_Res34','Elite360M_Res50'],
                    help="backbone")
parser.add_argument("--task", type=str, default="depth",
                    choices=['depth', 'segmentation', 'normal', 'depth_segmentation', 'depth_normal',
                             'normal_segmentation', 'depth_normal_segmentation'],
                    help="select task to evaluate")
parser.add_argument("--dataset_rootdir", default=None, type=str,help="root dir of dataset")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

parser.add_argument("--batch_size", type=int, default=1, help="batch size")

# loading and logging setting
parser.add_argument("--load_weights_dir", default=None, type=str,
                    help="folder of models to load")  # , default='./tmp/panodepth/models/weights_pretrain'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_stan"),
                    help="log directory")
parser.add_argument("--log_frequency", type=int, default=300, help="number of batches between each tensorboard log")
# dataset settings
parser.add_argument("--dataset", default='S2D3D', type=str, choices=['S2D3D', 'Struc3D','MP3D'],
                    help="Training dataset")

args = parser.parse_args()


def main():
    tester = Evaluator_(args)
    tester.validate()


if __name__ == "__main__":
    main()
