import os.path
import cv2
from PIL import Image
import numpy as np
number=0
root = "/home/ps/data/dataset/360Image/Dataset/Structured3D/"
f1 = open("/split_struc3d/structured3d_mul_train.txt", 'a+')
f2 = open("/split_struc3d/structured3d_mul_val.txt", 'a+')
f3 = open("/split_struc3d/structured3d_mul_test.txt", 'a+')

key=-1
for i in os.listdir(root):
    scene_id = int(i.split("_")[-1])

    if scene_id < 3000:
        split = "train"
    elif 3000 <= scene_id < 3250:
        split = "val"
    else:
        split = "test"
    sub_dir = os.path.join(root,i,"2D_rendering")
    for j in os.listdir(sub_dir):
        sub_sub_dir = os.path.join(sub_dir,j,"panorama","full")
        if "rgb_rawlight.png" in os.listdir(sub_sub_dir) and "depth.png" in os.listdir(sub_sub_dir) and "semantic.png" in os.listdir(sub_sub_dir) and  "normal.png" in os.listdir(sub_sub_dir):
            for k in os.listdir(sub_sub_dir):
                if "rgb_rawlight" in k:
                    rgb_name = os.path.join(sub_sub_dir,k)
                    try:
                        Image.open(rgb_name).tobytes()
                    except:
                        key=0
                        print(rgb_name)
                if "depth" in k:
                    depth_name = os.path.join(sub_sub_dir,k)
                    try:
                        Image.open(depth_name).tobytes()
                    except:
                        key = 0
                        print(depth_name)
                elif "semantic" in k:
                    semantic_name = os.path.join(sub_sub_dir,k)
                    try:
                        Image.open(semantic_name).tobytes()
                    except:
                        key = 0
                        print(semantic_name)

                elif "normal" in k:
                    normal_name = os.path.join(sub_sub_dir,k)
                    try:
                        Image.open(normal_name).tobytes()
                    except:
                        key = 0
                        print(normal_name)
        else:
            key = 0
            print(sub_sub_dir)
        # if key!=0:
        #     if split=="train":
        #         f1.write(rgb_name + " " + depth_name + " " + semantic_name + " " + normal_name)
        #         f1.write("\n")
        #     elif split=="val":
        #         f2.write(rgb_name + " " + depth_name + " " + semantic_name + " " + normal_name)
        #         f2.write("\n")
        #     elif split=="test":
        #         f3.write(rgb_name + " " + depth_name + " " + semantic_name + " " + normal_name)
        #         f3.write("\n")
        #     else:
        #         raise ValueError("no split")
        key=-1

print(number)
