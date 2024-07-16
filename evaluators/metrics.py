import torch
from sklearn.metrics import pairwise


# The next two methods were taken from https://github.com/ssnl/align_uniform
def align_loss(x, y, alpha=2):
    """
    x   : Tensor, shape=[bsz, d],
          latents for one side of positive pairs
    y   : Tensor, shape=[bsz, d]
          latents for the other side of positive pairs
          bsz : batch size (number of positive pairs)
          d   : latent dim
    """

    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    """
    x   : Tensor, shape=[bsz, d],
          latents for one side of positive pairs
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def cosine_similarity(x, y):
    """
    x   : Tensor, shape=[bsz, d],
          latents for one side of positive pairs
    y   : Tensor, shape=[bsz, d]
          latents for the other side of positive pairs
          bsz : batch size (number of positive pairs)
          d   : latent dim
    """
    pos_cosine = torch.nn.functional.cosine_similarity(x, y, dim=1).mean()
    neg_cosine = pairwise.cosine_similarity(x.cpu(), x.cpu()).mean()
    return pos_cosine, neg_cosine
