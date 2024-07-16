# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Updated for the needs of the project (rm a lot of stuff)

from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from models.ssl_models.ssl import Encoder
from models.ssl_models.utils import load_ssl_models
from evaluators.evaluate import Attack

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class DeACL(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        attack: Attack,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        scheduler: str,
        warmup_epochs: float = 10,
        warmup_start_lr: float = 0.003,
        trades_k: float = 3,
        min_lr: float = 0,
        extra_optimizer_args: Dict = dict(),
        lr_decay_steps: Sequence = None,
        **kwargs,
    ):
        """
        Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            model_cfg (dict): configuration of the model (teacher and student)
            attack (Attack): attack to use for adversarial training.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.
        """

        super().__init__()

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs

        # all the other parameters
        self.extra_args = kwargs

        self.backbone: Encoder = load_ssl_models(
            model_cfg, load_weights=True, ckpt_path=model_cfg.ckpt.final_ckpt
        ).encoder
        self.momentum_backbone: Encoder = load_ssl_models(
            model_cfg, load_weights=True, ckpt_path=model_cfg.ckpt.final_ckpt
        ).encoder
        self.momentum_backbone.freeze()

        self.attack = attack

        # DeACL-specific
        self.trades_k = trades_k

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, self.max_epochs, eta_min=self.min_lr
            )
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the offline momentum_backbone.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            out (torch.Tensor): the output of the momentum backbone (features)
        """
        out = self.momentum_backbone.get_representation(X)

        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """
        self.momentum_backbone.eval()

        image_weak = batch[0]

        ############################################################################
        # Adversarial Training (CAT)
        ############################################################################

        away_target = self.momentum_forward(image_weak)

        AE_generation_image = image_weak
        self.backbone.eval()
        image_AE, _ = self.attack.run(
            images=AE_generation_image,
            model=self.backbone,
            precomputed_original_representations=away_target,
        )
        self.backbone.train()

        image_CAT = torch.cat([image_weak, image_AE])
        logits_all = self.backbone.get_representation(image_CAT)
        bs = image_weak.size(0)
        student_logits_clean = logits_all[:bs]
        student_logits_AE = logits_all[bs:]

        # Cosine Similarity loss
        loss_first = -F.cosine_similarity(student_logits_clean, away_target).mean()
        loss_second = (
            -self.trades_k
            * F.cosine_similarity(student_logits_AE, student_logits_clean).mean()
        )
        adv_loss = loss_first + loss_second

        ############################################################################
        # Adversarial Training (CAT)
        ############################################################################

        metrics = {
            "adv_loss_total": adv_loss,
            "adv_loss_first": loss_first,
            "adv_loss_second": loss_second,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=False)

        return adv_loss
