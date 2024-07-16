# https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ssl_models.ssl import SSL_MODEL


class SimCLR(SSL_MODEL):
    """
    Build a SimCLR model.
    """

    def __init__(self, cfg):
        """
        dim: feature dimension (default: 2048)
        proj_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimCLR, self).__init__(cfg)
        self.temperature = cfg.temperature
        self.proj_hidden_dim = cfg.proj_hidden_dim
        self.proj_output_dim = cfg.proj_output_dim
        self.proj_batch_norm = cfg.proj_batch_norm

        # build a projector
        if cfg.proj_batch_norm:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.proj_hidden_dim, bias=False),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
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

    def info_nce_loss(self, z1, z2, device):
        features = torch.cat([z1, z2], dim=0)
        bs = z1.shape[0]
        n_views = 2
        labels = torch.cat([torch.arange(bs) for _ in range(n_views)], dim=0)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (n_views * bs, n_views * bs)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        return logits, labels
