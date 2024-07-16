# adapted from https://github.com/vturrisi/solo-learn/blob/main/solo/losses/simsiam.py
import torch.nn as nn
import torch.nn.functional as F
from models.ssl_models.ssl import SSL_MODEL


class SimSiam(SSL_MODEL):
    def __init__(self, cfg):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        self.proj_hidden_dim = cfg.proj_hidden_dim
        self.proj_output_dim = cfg.proj_output_dim
        self.pred_hidden_dim = cfg.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            nn.BatchNorm1d(self.proj_output_dim, affine=False),
        )
        # hack: not use bias as it is followed by BN
        self.projector[6].bias.requires_grad = False

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_output_dim, self.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(self.pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.pred_hidden_dim, self.proj_output_dim),
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

        z1 = self.projector(h1)  # NxC
        z2 = self.projector(h2)  # NxC

        p1 = self.predictor(p1)  # NxC
        p2 = self.predictor(p2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    def simsiam_loss(self, p1, p2, z1, z2):
        return (
            -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean())
            * 0.5
        )

    def get_representations(self, x, with_projection=False, with_prediction=False):
        h = self.encoder(x)
        z = self.projector(h)
        p = self.predictor(z)
        if with_projection:
            if with_prediction:
                return h, z, p
            return h, z
        return h
