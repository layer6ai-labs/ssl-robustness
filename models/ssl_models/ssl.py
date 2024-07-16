import torch
from typing import Optional, Tuple, Union
from models.backbones import get_encoder
from torch import Tensor as T


class SSL_MODEL(torch.nn.Module):
    name: str
    encoder: torch.nn.Module
    model: Optional[torch.nn.Module]  # for MAE
    projector: Optional[torch.nn.Module]
    predictor: Optional[torch.nn.Module]

    def __init__(self, cfg):
        super().__init__()
        self.name = cfg.name
        self.feature_dim = cfg.feature_dim
        self.encoder_arch = cfg.encoder_arch
        self.encoder = Encoder(cfg)

    def get_representation(
        self, x: T, with_projection=False, with_prediction=False
    ) -> Union[T, Tuple[T, T], Tuple[T, T, T]]:
        raise NotImplementedError

    def get_patch_embeddings(self, x: T) -> T:
        return self.encoder.get_patch_embeddings(x)


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.name = f"{cfg.name}_encoder"
        self.feature_dim = cfg.feature_dim
        self.encoder_arch = cfg.encoder_arch
        self.image_size = cfg.image_size
        if "patch_size" in cfg:
            self.patch_size = cfg.patch_size
        self.model = get_encoder(cfg.encoder_arch, cfg=cfg)

    def get_representation(self, x: T) -> T:
        return self.model(x)

    def forward(self, x: T) -> T:
        return self.model(x)

    def get_patch_embeddings(self, x: T) -> T:
        if getattr(self.model, "get_patch_embeddings", None) is None:
            raise ValueError(
                f"No patch embedding available for the backbone encoder {type(self.model)}"
            )
        return self.model.get_patch_embeddings(x)

    def parameters(self):
        return self.model.parameters()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
