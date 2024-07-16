# Models

Core directory containing implementations of different encoders based on different architectures and training paradigms.

## Supported encoders

* DINOv2 (from huggingface)
* DINOv1 (from huggingface, from checkpoint, ViT-Small, ViT-Tiny)
* SimSiam (from checkpoint, ResNet50, ViT-Tiny)
* SimCLR (from checkpoint, ResNet50, ResNet18, ViT-Tiny)

## configuration/

Contains *.json files for transformers.ViTModel configuration to support all ViT classes: Tiny, Small, Base, Large

## dino.py

Contains implementation of all DINO classes: v2, v1, v1OursSmall/Tiny (from checkpoints)

## downstream.py

Contains implementation of linear heads for downstream tasks: classification (LinearClassifier) and segmentation (SegmenterHead, Segmenter)

## encoder.py

Contains implementation of general Encoder class, inherited by all encoder classes

## fb_vit.py

Contains code copied from [Meta's DINO repo](https://github.com/facebookresearch/dino) used to load DINO checkpoints

## simclr.py

Contains implementation of SimCLR encoders for different architectures

## simsiam.py

Contains implementation of SimSiam encoders for different architectures

## vit.py

Implements helper `get_vit` function used to load ViT models

## Depth Estimation

Our code so far support linear heads for depth estimation. You can either train your own head or load it from [dinov2_vits14_nyu](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear_head.pth) or [inov2_vitb14_nyu](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_linear_head.pth). Make sure you put them under `tmp/depth_estimators/` and set `load_offical` in `conf/task/depth_estimation` to True.