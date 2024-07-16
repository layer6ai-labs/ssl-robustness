import json
import os

from transformers import ViTConfig, ViTModel


def get_transformer_vit(encoder_name, cfg, **kwargs):
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configuration",
        f"{encoder_name}.json",
    )
    with open(config_path, "r") as f_obj:
        config = json.load(f_obj)
    if cfg.override_dim is not None:
        config["hidden_size"] = cfg.override_dim
    return ViTModel(ViTConfig(**config))
