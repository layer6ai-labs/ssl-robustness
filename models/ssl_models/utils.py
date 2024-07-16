import torch
from models.ssl_models.simclr import SimCLR
from models.ssl_models.simsiam import SimSiam
from models.ssl_models.dino import DINOv1, DINOv2
from models.ssl_models.mae import MAE
from models.ssl_models.ssl import SSL_MODEL

from omegaconf import DictConfig

SSL_MODELS = {
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "dinov1": DINOv1,
    "dinov2": DINOv2,
    "mae": MAE,
}


def load_ssl_models(cfg, load_weights=False, ckpt_path=None) -> SSL_MODEL:
    model = SSL_MODELS[cfg.ssl_model](cfg)
    if load_weights:
        if ckpt_path is not None:
            model = load_weights_from_ckpt(model, ckpt_path, cfg)
        elif cfg["pretrained_url"] is not None:
            # only load encoder from torch hub
            # e.g. torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            model.encoder = torch.hub.load(
                cfg.pretrained_url.repo, cfg.pretrained_url.name
            )
        else:
            raise ValueError("Must provide ckpt_path or pretrained url to load weights")

    return model


def load_weights_from_ckpt(ssl_model, ckpt_path, cfg):
    encoder_name = f"{cfg.ckpt.encoder_name}." if len(cfg.ckpt.encoder_name) > 0 else ""
    projector_name = (
        f"{cfg.ckpt.projector_name}." if "projector_name" in cfg.ckpt else None
    )
    predictor_name = (
        f"{cfg.ckpt.predictor_name}." if "predictor_name" in cfg.ckpt else None
    )
    with open(
        ckpt_path,
        "rb",
    ) as f_obj:
        checkpoint = torch.load(f_obj, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "state_dict_key" in cfg.ckpt and cfg.ckpt.state_dict_key in checkpoint:
        state_dict = checkpoint[cfg.ckpt.state_dict_key]
    else:
        state_dict = checkpoint

    encoder_state_dict, projector_state_dict, predictor_state_dict = {}, {}, {}
    for k in list(state_dict.keys()):
        # some ckpts have total_ops and total_params with thop
        if "total_ops" in k or "total_params" in k:
            continue
        # retain only encoder up to before the embedding layer
        if k.startswith(encoder_name):
            # remove prefix
            encoder_state_dict[k[len(encoder_name) :]] = state_dict[k]
        elif projector_name is not None and k.startswith(projector_name):
            projector_state_dict[k[len(projector_name) :]] = state_dict[k]
        elif predictor_name is not None and k.startswith(predictor_name):
            predictor_state_dict[k[len(predictor_name) :]] = state_dict[k]

    if "non_standard_naming" in cfg.ckpt and cfg.ckpt.non_standard_naming:
        encoder_state_dict = modify_state_dict_naming(
            ssl_model.encoder.model.state_dict(), encoder_state_dict
        )
    if "drop_fc" in cfg.ckpt and cfg.ckpt.drop_fc:
        for k in list(encoder_state_dict.keys()):
            if "fc" in k:
                del encoder_state_dict[k]

    if "drop_norm" in cfg.ckpt and cfg.ckpt.drop_norm:
        del encoder_state_dict["normalize.mean"]
        del encoder_state_dict["normalize.std"]

    ssl_model.encoder.model.load_state_dict(encoder_state_dict, strict=True)
    if projector_name is not None:
        ssl_model.projector.load_state_dict(projector_state_dict, strict=False)
    if predictor_name is not None:
        ssl_model.predictor.load_state_dict(predictor_state_dict, strict=False)
    return ssl_model


def modify_state_dict_naming(current_model_kvpair, pretrained_model_kvpair):
    new_state_dict = {}
    pretrained = list(pretrained_model_kvpair.items())
    count = 0
    for current_name, current_weights in current_model_kvpair.items():
        if "fc" in current_name and count >= len(pretrained):
            # skip the last fc layer
            break
        pretrained_name, pretrained_weights = pretrained[count]
        assert (
            current_weights.shape == pretrained_weights.shape
        ), f"Shape mismatch: {current_name} of shape {current_weights.shape} \
                and {pretrained_name} of shape {pretrained_weights.shape}"
        new_state_dict[current_name] = pretrained_weights.detach().clone()
        count += 1
    return new_state_dict


def encoder_sanity_check(
    ssl_model_cfg: DictConfig,
    path: str,
    device: torch.device,
    trained_model,
):
    encoder = load_ssl_models(
        ssl_model_cfg, load_weights=True, ckpt_path=path
    ).encoder.to(device)
    for k, v in encoder.state_dict().items():
        assert torch.equal(
            v, trained_model.encoder.state_dict()[k]
        ), f"Encoder not loaded correctly at {k}"
