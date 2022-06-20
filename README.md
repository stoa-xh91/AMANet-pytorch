# AMANet: Adaptive Multi-Path Aggregation for Learning Human 2D-3D Correspondences

# Introduction
This is the implementation of AMANet: Adaptive Multi-Path Aggregation for Learning Human 2D-3D Correspondences. 

![](/figures/framework.jpg)
# Environment
The code is developed based on the [Detectron2](https://github.com/facebookresearch/detectron2) platform. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards or Tesla V100 cards. Other platforms or GPU cards are not fully tested.
# Installation
## Requirements
- Linux with Python=3.7
- Pytorch = 1.4 and torchvision that matches the Pytorch installation. Please install them together at [pytorch.org](https://pytorch.org/)
- OpenCV is needed by demo and visualization
- We recommend using **anaconda3** for environment management

## Build detectron2 from AMA
```
git clone https://github.com/stoa-xh91/AMANet-pytorch
cd AMANet-pytorch/
python -m pip install -e .
```

# Prepare

## Data prepare


1. Request dataset here: [DensePose](https://github.com/facebookresearch/detectron2)

2. Please download dataset under datasets

Make sure to put the files as the following structure:

```
  ├─configs
  ├─datasets
  │  ├─coco
  │  │  ├─annotations
  │  │  ├─train2014
  │  │  ├─val2014
  ├─demo
  ├─detectron2
```

# Training 
- Change the config file depending on what you want. Here, we provide a way to train KTN models
```
# Example: training AMANet with ResNet-50 backbone on DensePose-COCO with GPU 0 1 2 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/AMA/train_net.py \
--num-gpus 4 \
--config-file projects/AMA/configs/densepose_AMA_R_50_FPN_s1x.yaml \
OUTPUT_DIR work_dirs/densepose_AMA_R_50
```
After training, the final model is saved in OUTPUT_DIR.

# Testing
- To test the trained models saved in <work_dir>, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python projects/AMA/train_net.py \
--num-gpus 4 \
--config-file projects/AMA/configs/densepose_AMA_R_50_FPN_s1x.yaml \
--eval-only \
MODEL.WEIGHTS work_dirs/densepose_AMA_R_50/model_final.pth
```

# Acknowledge
Our code is mainly based on [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose). 

# Citation

```
@article{WangAMA++,  
	title={AMANet: Adaptive Multi-Path Aggregation for Learning Human 2D-3D Correspondences},
	author={Wang, Xuanhan and Gao, Lianli and Song, Jingkuan and Guo, Yuyu and Shen, Heng Tao},  
	journal={IEEE Transactions on Multimedia},   
	year={2021}
}

@inproceedings{densepose:amanet,
	title={Adaptive Multi-Path Aggregation for Human DensePose Estimation in the Wild},
	author={Guo, Yuyu and Gao, Lianli and Song, Jingkuan and Wang, Peng and Xie, Wuyuan and Shen, Heng Tao},
	pages={356--364},
	booktitle = {ACM MM},
	year={2019}
}

```