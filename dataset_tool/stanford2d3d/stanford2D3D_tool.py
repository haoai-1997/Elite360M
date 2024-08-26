import os
import glob
import argparse
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import rescale
import json
from PIL import Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ori_root', required=True)
parser.add_argument('--new_root', required=True)
args = parser.parse_args()

areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']

with open('semantic_labels.json') as f:
    semantic_id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']

with open('name2label.json') as f:
    semantic_name2id = json.load(f)

colors = np.load('colors.npy')

id2label = np.array([semantic_name2id[name] for name in semantic_id2name], np.uint8)

for area in areas:
    print('Processing:', area)
    os.makedirs(os.path.join(args.new_root, area, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'semantic'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.new_root, area, 'semantic_visualize'), exist_ok=True)
    for fname in tqdm(os.listdir(os.path.join(args.ori_root, area, 'pano', 'rgb'))):
        if fname[0] == '.' or not fname.endswith('png'):
            continue
        rgb_path = os.path.join(args.ori_root, area, 'pano', 'rgb', fname)
        d_path = os.path.join(args.ori_root, area, 'pano', 'depth', fname[:-7] + 'depth.png')
        sem_path = os.path.join(args.ori_root, area, 'pano', 'semantic', fname[:-7] + 'semantic.png')
        norm_path = os.path.join(args.ori_root, area, 'pano', 'normal', fname[:-7] + 'normals.png')

        assert os.path.isfile(d_path)
        assert os.path.isfile(sem_path)
        assert os.path.isfile(norm_path)

        rgb = imageio.v2.imread(rgb_path, pilmode='RGB')[..., :3]
        rgb = rescale(rgb, 0.25, order=0, mode='wrap', anti_aliasing=False, preserve_range=True, channel_axis=2)

        depth = imageio.v2.imread(d_path)
        depth = rescale(depth, 0.25, order=0, mode='wrap', anti_aliasing=False, preserve_range=True)

        sem = np.array(Image.open(sem_path).resize((1024, 512), Image.NEAREST), np.int32)
        unk = (sem[..., 0] != 0)
        sem = id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0

        normal = np.array(Image.open(norm_path).convert("RGB").resize(size=(1024, 512),
                                                                      resample=Image.NEAREST)).astype(np.uint8)

        vis = np.array(rgb)
        vis = vis // 2 + colors[sem] // 2
        Image.fromarray(vis).save(
            os.path.join(args.new_root, area, 'semantic_visualize', fname[:-7] + 'semantic_visualize.png'))

        imageio.v2.imwrite(os.path.join(args.new_root, area, 'rgb', fname), rgb.astype(np.uint8))
        imageio.v2.imwrite(os.path.join(args.new_root, area, 'depth', fname[:-7] + 'depth.png'),
                           depth.astype(np.uint16))
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1
        Image.fromarray(sem).save(os.path.join(args.new_root, area, 'semantic', fname[:-7] + 'semantic.png'))
        Image.fromarray(normal).save(os.path.join(args.new_root, area, 'normal', fname[:-7] + 'normals.png'))
