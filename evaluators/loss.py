import torch
from typing import Tuple


class Loss:
    name: str
    loss_func = torch.nn.Module
    loss: torch.nn.Module

    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, true)

    def match_shapes(
        self, benign_repr: torch.Tensor, adv_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if list(benign_repr.shape) == list(adv_repr.shape):
            return benign_repr, adv_repr
        else:
            num_steps = adv_repr.shape[1]
            return benign_repr.unsqueeze(1).repeat(1, num_steps, 1).permute(
                0, 2, 1
            ), adv_repr.permute(0, 2, 1)

    def get_distances(
        self, benign_repr: torch.Tensor, adv_repr: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        if normalize:
            benign_repr, adv_repr = self.normalize_representations(
                benign_repr, adv_repr
            )
        benign_dist = self.loss_func(reduction="none")(benign_repr, benign_repr)
        adv_dist = self.loss_func(reduction="none")(
            *self.match_shapes(benign_repr, adv_repr)
        )
        try:  # for crossentropy
            benign_dist = benign_dist.mean(dim=1)
            adv_dist = adv_dist.mean(dim=1)
        except:
            pass
        benign_dist = benign_dist.view(-1, 1)
        if len(adv_dist.shape) == 1:
            adv_dist = adv_dist.view(-1, 1)
        return torch.concatenate((benign_dist, adv_dist), dim=1)

    def normalize_representations(
        self, benign_repr: torch.Tensor, adv_repr: torch.Tensor
    ) -> torch.Tensor:
        mean, std = benign_repr.mean(dim=1).unsqueeze(1), benign_repr.std(
            dim=1
        ).unsqueeze(1)
        benign_repr = (benign_repr - mean) / std
        if len(adv_repr.shape) == 3:
            mean, std = mean.unsqueeze(2), std.unsqueeze(2)
        adv_repr = (adv_repr - mean) / std
        return benign_repr, adv_repr


class L2(Loss):
    name = "L2"
    loss_func = torch.nn.MSELoss
    loss = torch.nn.MSELoss()


class Cosine(Loss):
    name = "cosine"
    loss = torch.nn.CosineSimilarity()

    def __call__(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return -self.loss(true, pred).sum()

    def get_distances(
        self, benign_repr: torch.Tensor, adv_repr: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        if normalize:
            benign_repr, adv_repr = self.normalize_representations(
                benign_repr, adv_repr
            )
        benign_dist = -self.loss(benign_repr, benign_repr).view(-1, 1)
        adv_dist = -self.loss(*self.match_shapes(benign_repr, adv_repr))
        if len(adv_dist.shape) == 1:
            adv_dist = adv_dist.view(-1, 1)
        return torch.concatenate((benign_dist, adv_dist), dim=1)


class CrossEntropy(Loss):
    name = "crossentropy"
    loss_func = torch.nn.CrossEntropyLoss
    loss = torch.nn.CrossEntropyLoss()

    def normalize_representations(
        self, benign_repr: torch.Tensor, adv_repr: torch.Tensor
    ) -> torch.Tensor:
        s = torch.nn.Softmax(dim=-1)
        return s(benign_repr), s(adv_repr)


class Mismatch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        true_logits = pred[torch.arange(len(pred)), true].clone()
        pred[torch.arange(len(pred)), true] = -torch.inf
        target_logits = pred.max(dim=1).values
        return (target_logits - true_logits).sum()


class MismatchLoss(Loss):
    name = "mismatch"
    loss = Mismatch()
    loss_func = Mismatch


loss_dict = {
    "L2": L2,
    "cosine": Cosine,
    "crossentropy": CrossEntropy,
    "mismatch": MismatchLoss,
}
