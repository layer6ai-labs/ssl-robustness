import torch.nn as nn
from models.ssl_models.ssl import SSL_MODEL
from models.backbones.utils import trunc_normal_
import torch


class DINOv1(SSL_MODEL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj_output_dim = cfg.proj_output_dim
        self.use_bn_in_head = cfg.use_bn_in_head
        self.projector = DINOv1Head(
            self.feature_dim, self.proj_output_dim, self.use_bn_in_head
        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            z1, z2
        """

        # compute features for one view
        h1 = self.encoder(x1)  # NxC
        h2 = self.encoder(x2)  # NxC

        z1 = self.predictor(h1)  # NxC
        z2 = self.predictor(h2)  # NxC

        return z1, z2

    def get_representations(self, x, with_projection=False):
        h = self.encoder(x)
        if with_projection:
            z = self.projector(h)
            return h, z
        return h


class DINOv2(SSL_MODEL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj_output_dim = cfg.proj_output_dim
        self.use_bn_in_head = cfg.use_bn_in_head
        self.projector = DINOv2Head(
            self.feature_dim, self.proj_output_dim, self.use_bn_in_head
        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            z1, z2
        """

        # compute features for one view
        h1 = self.encoder(x1)  # NxC
        h2 = self.encoder(x2)  # NxC

        z1 = self.predictor(h1)  # NxC
        z2 = self.predictor(h2)  # NxC

        return z1, z2

    def get_representations(self, x, with_projection=False):
        h = self.encoder(x)
        if with_projection:
            z = self.projector(h)
            return h, z
        return h


class DINOv1Head(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOv2Head(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
