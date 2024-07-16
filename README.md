# SSL-Robustness

This is the codebase for benchmarking adversarial robustness of Self-Supervised Learning across different downstream tasks presented in [paper name](arxiv link) and accepted at [ICML 2024 Workshop FM-Wild](https://openreview.net/forum?id=U2nyqFbnRF).

## Installation & Usage

Install python 10
```
conda create -n robust-ssl python=3.10
conda activate robust-ssl
```
Install PyTorch 1.13.0
```
conda install pytorch=1.13.0 torchvision pytorch-cuda=[YOUR_CUDA_VERSION_HERE] -c pytorch -c nvidia
```
Install the rest of requirements
```
conda install --file requirements.txt
conda install -c conda-forge timm
pip install hydra-core --upgrade
pip install -U albumentations
pip install -U datasets
pip install git+https://github.com/fra31/auto-attack
pip install pytorch-lightning lightning-bolts wandb
```
To better organize the paramerters, we use [hydra](hydra.cc). For now, we have three groups of configuration. 1. action, 2. task, and 3.ssl_models. You can update the hyperparameters for each action and task inside their corresponding yaml file.

To run main method you should pass which action and which task you want to do, and which ssl encoder you would like to use.

```
python main.py +action=$ACTION training +task=$TASK +ssl_model=$ENCODER task.dataset=$DATASET
```

You can run this code for different actions. Here are different actions and what they do:

`downstream_training`: trains a downstream model (segmenter, classifier or depth estimator) on top of the encoder and reports the validation accuracy

`adversarial_attack`: attacks both the encoder and the downstream model and reports the adversarial accuracy. For segmentation for now we attack only the encoder and report adversarial mean accuracy and mean IoU.

For `task` you can selection one of three tasks: `classification`, `segmentation`, `depth_estimation`. For `ssl_model`, you can select any of the files inside `conf/ssl_model/`, just remove `.yaml` suffix. Finally, for the `task.dataset`, you can select any dataset list [here](https://github.com/atiyeh-ashari/robust-ssl-l6/blob/atiyeh/publish/data/local_datasets.py#L456-L468).

## Checkpoints
Please download the checkpoints folder from this [drive](https://drive.google.com/drive/folders/11vS0t1BDQAFoqGWjBsyFuffbR_2gQihS?usp=sharing) and put it under [tmp/](tmp/) to get started. 

### Checkpoints/SSL models that we currently support:

| Pretrained| SSL Model | Finetuned?| Backbone |Config| Source |
|-----------|--------|--------------------|----------|----------|--------|
| CIFAR10   | SimCLR  | No   |ResNet-50   |[simclr_cifar10_resnet50](conf/ssl_model/simclr_cifar10_resnet50.yaml)            |[Pytorch SimCLR](https://github.com/echoyi/SimCLR) |
| CIFAR10   | SimCLR  | No   |ResNet-18 |[simclr_cifar10_resnet18_solo](conf/ssl_model/simclr_cifar10_resnet18_solo.yaml)     |[solo-learn](https://github.com/vturrisi/solo-learn) |
| CIFAR10   | SimCLR  | DeACL|ResNet-18 |[simclr_cifar10_resnet18_DeACL](conf/ssl_model/simclr_cifar10_resnet18_DeACL.yamll)   | Finetuned with DeACL from solo-learn checkpoint|
| CIFAR10   | SimSiam | No   |ResNet-18 |[simsiam_cifar10_resnet18_solo](conf/ssl_model/simsiam_cifar10_resnet18_solo.yaml)    |[solo-learn](https://github.com/vturrisi/solo-learn) |
| ImageNet  | SimCLRv1  | No   |ResNet-50   |[simclr_v1_imagenet_resnet50](conf/ssl_model/simclr_v1_imagenet_resnet50_1x.yaml)            |[How Well Do Self-Supervised Models Transfer? repo](https://github.com/linusericsson/ssl-transfer) + [SimCLRv1 tf to torch converter](https://github.com/tonylins/simclr-converter)|
| ImageNet  | SimCLRv2  | No   |ResNet-50   |[simclr_v2_imagenet_resnet50](conf/ssl_model/simclr_v2_imagenet_resnet50_1x.yaml)            |[How Well Do Self-Supervised Models Transfer? repo](https://github.com/linusericsson/ssl-transfer) + [SimCLRv2 tf to torch converter](https://github.com/Separius/SimCLRv2-Pytorch)|
| ImageNet  | DINOv1  | No   |ViT-small (patch size 8) |[dino_v1_vits8]([conf/ssl_model/dino_v1_vits8.yaml)      |[dino](https://github.com/facebookresearch/dino) |
| ImageNet  | DINOv1  | No   |ViT-small (patch size 16) |[dino_v1_vits16]([conf/ssl_model/dino_v1_vits16.yaml)      |[dino](https://github.com/facebookresearch/dino) |
| ImageNet  | DINOv1  | No   |ViT-tiny (patch size 16) |[dino_v1_vitt16_ours]([conf/ssl_model/dino_v1_vits16.yaml) |Trained by SprintML|
| ImageNet  | DINOv1  | Adversarial training   |ViT-small (patch size 16) |[dino_v1_vits16_adv_ours]([conf/ssl_model/dino_v1_vits16_adv_ours.yaml) |Trained by SprintML|
| DINO      | DINOv2  | No |ViT-small (patch size 16) |[dino_v2_vits14_distilled]([conf/ssl_model/dino_v2_vits14_distilled.yaml) |[dinov2](https://github.com/facebookresearch/dinov2)|

## Adding a SSL MODEL
To add a SSL MODEL:
create a new config for the model under the config folder [/conf/ssl_model/](/conf/ssl_model/) with the required parameters. For example, to add dino_vitb16 from the [dino repo](https://github.com/facebookresearch/dino), create a file "dino_v1_vitb16.yaml" under [/conf/ssl_model/](/conf/ssl_model/) as following:
```
name: "dino_v1_vitb16" # The name of the Pretrained SSL model
ssl_model: "dinov1" # Type of SSL model, see models/ssl_models/utils.py for a full list
encoder_arch: "vitb" 
proj_output_dim: 65536
use_bn_in_head: False
feature_dim: 768 
num_heads: 12 
patch_size: 16 
image_size: 224
pretrained_url:
  repo: "facebookresearch/dino:main"
  name: "dino_vitb16"
ckpt:
  ckpts_folder: null 
  final_ckpt: "tmp/checkpoints/imagenet/dino/official/dino_vitb16.ckpt" 
  encoder_name: "backbone"
  projector_name: "head"
  non_standard_naming: False
  state_dict_key: "teacher"
```

There are two ways to specify the pretrained checkpoint paths:
- TO examine only one checkpoint (usually the final checkpoint) of the trained SSL model, then pass that full file path as in `final_ckpt` under `ckpt`.
- If you would like to examine more than one checkpoints of an encoder (such as a training history), pass the folder's name as `ckpts_folder` under `ckpt`. The code will load each file under the folder. (Make sure there is no other file rather than checkpoints to be tested in the folder)

## Datasets

The code supports the following datasets: cifar100, imagenet, foodseg103, cifar100, mnist, fashion-mnist, flowers102, food101, stl10, ADE20K, CityScapes, Pascal VOC 2012, NYU Depth v2. For the latter ones you need to take the actions below to download/parse the datasets.

### ADE20k

We use ADE20k from the [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)

To load ADE20k:

1. first download data from [here](http://sceneparsing.csail.mit.edu/) (link→downloads→Scene Parsing→Data:[train/val(922MB)]) 
2. insert the zip file into `tmp/data/ade20k/`
3. unzip it
4. run `python3 data/ade20k_to_hgdataset.py --path_to_ade20k tmp/data/ade20k/ADEChallengeData2016 --path_to_hgdataset tmp/data/ade20k_hg`


### CityScapes

We use CityScapes from the [Semantic Understanding of Urban Street Scenes](https://www.cityscapes-dataset.com/)

To load CityScapes:

1. first download data, `gt_Fine_trainvaltest.zip` and `leftImg8bit_trainvalset.zip` from [here](https://www.cityscapes-dataset.com/downloads/)
2. unzip both of them to one directory: `tmp/data/cityscapes` so it contains two folders: `gtFine` and `leftImg8bit`.
3. run `python3 data/cityscapes_to_hg_dataset.py`

### Pascal VOC 2012

We use Pascal VOC 2012 from the [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data)

To load Pascal VOC 2012:

1. first download data, `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar .`
2. `mkdir tmp/data/voc && tar -xvf VOCtrainval_11-May-2012.tar -C tmp/data/voc`
3. run `python3 data/pascal_voc_to_hg_dataset.py`

### NYU Depth V2

We use [HuggingFace NYU Depth V2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) dataset. It loads it automatically but takes several hours to load and process the images.

## Code formatting

This repo uses black for python formatting.
