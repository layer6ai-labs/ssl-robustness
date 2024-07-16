from models.backbones.resnet import get_resnet
from models.backbones.vision_transformer import get_vit
from models.backbones.mae_vision_transformers import (
    get_mae_vit_encoder,
    get_mae_vit_full,
)
from models.backbones.transformer_vit import get_transformer_vit


def get_encoder(encoder_name, cfg, **kwargs):
    if encoder_name.startswith("resnet"):
        return get_resnet(encoder_name, cfg, **kwargs)
    elif encoder_name.startswith("mae_vit"):
        return get_mae_vit_encoder(encoder_name, cfg, **kwargs)
    elif encoder_name.startswith("transformer-vit"):
        return get_transformer_vit(encoder_name, cfg, **kwargs)
    elif encoder_name.startswith("vit"):
        return get_vit(encoder_name, cfg, **kwargs)
    else:
        raise NotImplementedError


def get_encoder_decoder(model_name, cfg, **kwargs):
    assert model_name.startswith("mae_vit")
    return get_mae_vit_encoder(model_name, cfg, **kwargs)
