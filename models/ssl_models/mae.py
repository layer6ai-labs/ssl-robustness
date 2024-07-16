import torch.nn as nn
from models.backbones import get_encoder_decoder
from models.ssl_models.ssl import SSL_MODEL


class MAE(SSL_MODEL):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = get_encoder_decoder(self.encoder_arch, cfg=cfg)

    def get_representations(self, x, with_projection=False):
        h = self.encoder(x)
        if with_projection:
            return h, h
        return h

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
