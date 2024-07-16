import torchvision.models
import torch.nn as nn
import torch


def get_resnet(model_name, cfg, **kwargs):
    model = torchvision.models.__dict__[model_name](zero_init_residual=True)
    # remove fc
    model.fc = nn.Identity()
    if "cifarResNet" in cfg and cfg.cifarResNet:
        # solo repo uses padding of 2 but all other use padding of 1
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=cfg.conv1_padding, bias=False
        )
        model.maxpool = nn.Identity()
    return model
