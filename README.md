# Elite360M

Office source code of paper **Elite360M: Efficient 360 Multi-task Learning via Bi-projection Fusion and Cross-task Collaboration**, [Arxiv](https://arxiv.org/abs/2408.09336), [Project]()

# Preparation

#### Installation

Environments

* python 3.10
* Pytorch >= 1.12.0
* CUDA >= 11.3

Install requirements

```bash
pip install -r requirements.txt
```

# Training 

#### ResNet-18 as ERP branch encoder on Matterport3D

```
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29122 train_elite360m.py
--log_dir ./workdirs_uni/MP3D --gpu_devices 1 2 --backbone Elite360M_Res18 --batch_size 2 --dataset MP3D --dataset_rootdir $DATASET_DIR --num_epochs 150 --task depth_normal_segmentation --num_workers 8
```

It is similar for other datasets. 

# Evaluation  

```
CUDA_VISIBLE_DEVICES=1 python eval_elite360m.py --load_weights_dir ./workdirs_uni/MP3D/ --log_dir ./test_uni/MP3D --backbone Elite360M_Res18 --dataset MP3D --dataset_rootdir $DATASET_DIR --task depth_normal_segmentation
```

## Citation

Please cite our paper if you find our work useful in your research.

```
@article{ai2024elite360m,
  title={Elite360M: Efficient 360 Multi-task Learning via Bi-projection Fusion and Cross-task Collaboration},
  author={Ai, Hao and Wang, Lin},
  journal={arXiv preprint arXiv:2408.09336},
  year={2024}
}
```
# Acknowledgements

We thank the authors of the projects below:  
*[HexRUNet](https://github.com/matsuren/HexRUNet_pytorch)*,*[MultiPanoWise](https://github.com/Uzshah/MultiPanoWise)*,
*[Multi-Task-Transformer](https://github.com/prismformore/Multi-Task-Transformer)*,
*[MTL-Homoscedastic-Uncertainty
](https://github.com/hardianlawi/MTL-Homoscedastic-Uncertainty)*

If you find these works useful, please consider citing:
```
@inproceedings{zhang2019orientation,
  title={Orientation-aware semantic segmentation on icosahedron spheres},
  author={Zhang, Chao and Liwicki, Stephan and Smith, William and Cipolla, Roberto},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3533--3541},
  year={2019}
}
```
```
@inproceedings{invpt2022,
  title={Inverted Pyramid Multi-task Transformer for Dense Scene Understanding},
  author={Ye, Hanrong and Xu, Dan},
  booktitle={ECCV},
  year={2022}
}
```
```
@article{ye2023invpt++,
  title={InvPT++: Inverted Pyramid Multi-Task Transformer for Visual Scene Understanding},
  author={Ye, Hanrong and Xu, Dan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```
```
@inproceedings{shah2024multipanowise,
  title={MultiPanoWise: holistic deep architecture for multi-task dense prediction from a single panoramic image},
  author={Shah, Uzair and Tukur, Muhammad and Alzubaidi, Mahmood and Pintore, Giovanni and Gobbetti, Enrico and Househ, Mowafa and Schneider, Jens and Agus, Marco},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1311--1321},
  year={2024}
}
```
```
@inproceedings{kendall2018multi,
  title={Multi-task learning using uncertainty to weigh losses for scene geometry and semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7482--7491},
  year={2018}
}
```
