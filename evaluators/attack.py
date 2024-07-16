from typing import Tuple, Dict

import numpy as np
import torch
from evaluators.loss import Loss, CrossEntropy
from evaluators.utils import sliding_window_inference
from models.downstream import LinearClassifier, Segmenter, NormalizationWrapper
from models.ssl_models.ssl import Encoder

from typing import Callable
from torchvision import transforms

from autoattack import AutoAttack as AA

from models.downstream_models.depther.encoder_decoder import DepthEncoderDecoder
from evaluators.depth_losses.sigloss import SigLoss
from evaluators.depth_losses.gradientloss import GradientLoss


class Attack:
    name: str

    def __init__(
        self,
        loss: Loss,
        num_steps: int,
        lr: float,
        eps_budget: float,
        mean: tuple = None,
        std: tuple = None,
    ):
        self.loss = loss
        self.num_steps = num_steps
        self.lr = self._cast_to_float(lr)
        self.eps_budget = self._cast_to_float(eps_budget)
        self.left_boundary = 0
        self.right_boundary = 1
        self.mean = mean
        self.std = std

        self._normalize = transforms.Normalize(mean, std)
        self._denormalize = transforms.Normalize(
            -np.array(mean) / np.array(std), 1 / np.array(std)
        )

    def run(self, images, model, **kwargs):
        raise NotImplementedError

    def _cast_to_float(self, val) -> float:
        if not isinstance(val, float):
            try:
                return float(eval(val))
            except NameError:
                raise ValueError(
                    f"Value should be a float or a string that can be evaluated to a float (e.g. 1/255), got {val}"
                )
        else:
            return val


class PGD(Attack):
    name: str
    loss: Loss
    num_steps: int
    lr: float
    eps_budget: float
    optimizer: torch.optim.Optimizer = None
    delta: torch.Tensor = None

    def _init_delta_optim(self, images: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def _clamp_delta(self, images):
        # makes sure the delta is within budget and we don't get out of bounds of
        # [lb; rb] for images + delta
        self.delta.data = (
            torch.clamp_(
                images + self.delta.data,
                min=self.left_boundary,
                max=self.right_boundary,
            )
            - images
        )
        self.delta.data = torch.clamp_(
            self.delta.data, min=-self.eps_budget, max=self.eps_budget
        )

    def _optimize_delta(self):
        raise NotImplementedError

    def run(
        self,
        images: torch.Tensor,
        model: Encoder,
        precomputed_original_representations: torch.Tensor = None,
        override_delta: torch.Tensor = None,
        return_step_by_step: bool = False,
        steps_to_save: int = 100,
        get_representation: Callable = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        if get_representation is None:
            get_representation = model.get_representation
        if override_delta is None:  # useful for step-by-step evaluation of pgd
            self._init_delta_optim(images)
        if precomputed_original_representations is None:
            with torch.no_grad():
                original_representations = get_representation(self._normalize(images))
        else:
            original_representations = precomputed_original_representations

        if return_step_by_step:  # useful for step-by-step evaluation of pgd
            step_by_step_representations = []
            step_by_step_adv_images = []

        for idx in range(self.num_steps):
            adv_representations = get_representation(
                self._normalize(images + self.delta)
            )
            if return_step_by_step and not idx % int(
                self.num_steps / steps_to_save
            ):  # don't save all by default
                step_by_step_representations.append(
                    adv_representations.clone().detach().unsqueeze(0).cpu()
                )
                step_by_step_adv_images.append(
                    self._normalize(images + self.delta)
                    .clone()
                    .detach()
                    .unsqueeze(0)
                    .cpu()
                )
            self.loss(adv_representations, original_representations).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.delta.grad.zero_()

        adv_representations = get_representation(self._normalize(images + self.delta))

        if return_step_by_step:  # save final
            step_by_step_representations.append(
                adv_representations.clone().detach().unsqueeze(0).cpu()
            )
            step_by_step_adv_images.append(
                self._normalize(images + self.delta).clone().detach().unsqueeze(0).cpu()
            )

        adv_images: torch.Tensor = self._normalize(images + self.delta)
        if return_step_by_step:
            return (
                torch.concatenate(step_by_step_adv_images, dim=0),
                torch.concatenate(step_by_step_representations, dim=0),
            )
        return adv_images.detach(), adv_representations.detach()


class PGD_Sign(PGD):
    name = "sign"

    # Adapted from https://github.com/PKU-ML/DYNACL/blob/master/train_DynACL.py
    def _init_delta_optim(self, images: torch.Tensor):
        self.delta = torch.zeros_like(images).uniform_(
            -self.eps_budget, self.eps_budget
        )
        self.delta = self.delta.detach().requires_grad_(True)

    def _optimize_delta(self):
        self.delta.data += torch.sign(self.delta.grad) * self.lr


class DownstreamPGD(PGD_Sign):
    name = "downstream"
    loss = CrossEntropy()

    def __init__(self, num_steps, lr, eps_budget, loss=None, mean=None, std=None):
        super().__init__(self.loss, num_steps, lr, eps_budget, mean, std)

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: LinearClassifier,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        self._init_delta_optim(images)

        for _ in range(self.num_steps):
            logits = model(self._normalize(images + self.delta))
            with torch.no_grad():
                div = torch.maximum(logits, torch.ones_like(logits)).max(
                    dim=1, keepdim=True
                )[0]
            self.loss(logits / div, labels).backward()

            self._optimize_delta()
            self._clamp_delta(images)

            self.delta.grad.zero_()

        adv_images = self._normalize(images + self.delta)
        success = torch.argmax(model(adv_images), dim=1) != labels
        return adv_images.detach(), success.detach()


class AutoAttack(Attack):
    name = "autoattack"

    def __init__(self, eps_budget: float, mean=None, std=None):
        super().__init__(None, None, 0.0, eps_budget, mean, std)

    def run(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: LinearClassifier,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._denormalize(images)
        norm_model = NormalizationWrapper(model, self._normalize)
        attack = AA(
            norm_model,
            norm="Linf",
            eps=self.eps_budget,
            version="standard",
            verbose=False,
        )
        adv_images = attack.run_standard_evaluation(images, labels, bs=images.size(0))
        success = torch.argmax(norm_model(adv_images), dim=1) != labels
        adv_images = self._normalize(adv_images)
        return adv_images, success.detach()


# adapted based on https://github.com/asif-hanif/segpgd/blob/main/segpgd.py
class SegPGD(PGD):
    name = "SegPGD"

    def run(
        self,
        model: Segmenter,
        image: torch.Tensor,
        label: torch.Tensor,
        model_cfg: dict,
        task_cfg: dict,
        device=None,
        targeted=False,
        return_step_by_step=False,
    ):
        # model for volumetric image segmentation
        # images: [B,C,H,W]. B=BatchSize, C=Number-of-Channels,  H=Height,  W=Width
        # label: [B,1,H,W] (in integer form)

        adv_imgs = []
        img_size = model.encoder.image_size
        num_classes = model.head.num_labels

        image = self._denormalize(image).clone().detach().to(device)  #  [B,C,H,W]
        label = label.clone().detach().to(device)  #  [B,H,W]
        adv_image = image.clone().detach()

        # starting at a uniformly random point
        self.delta = torch.empty_like(adv_image).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self._clamp_delta(image)

        adv_image = adv_image + self.delta
        if return_step_by_step:
            adv_imgs.append(
                self._normalize(adv_image).clone().detach().unsqueeze(0).cpu()
            )

        for i in range(self.num_steps):
            adv_image.requires_grad = True
            # adv_image[B,C,H,W] --> adv_logits[B,NumClass,H,W]
            adv_logits = sliding_window_inference(
                model,
                num_classes=num_classes,
                input_data=self._normalize(adv_image),
                crop_size=(img_size, img_size),
                stride=(model_cfg.stride_size, model_cfg.stride_size),
                batch_size=task_cfg.batch_size,
                return_logits=True,
            )
            # [B,NumClass,H,W,D] --> [B,H,W]
            pred_label = torch.argmax(adv_logits, dim=1)
            # correctly classified voxels  [B,H,W]
            correct_voxels = label == pred_label
            # wrongly classified voxels    [B,H,W]
            wrong_voxels = label != pred_label
            # [B,NumClass,H,W] -->  [NumClass,B,H,W]
            adv_logits = adv_logits.permute(1, 0, 2, 3)
            # calculate loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_correct = loss_fn(
                adv_logits[:, correct_voxels].permute(1, 0),
                label[correct_voxels],
            )
            loss_wrong = loss_fn(
                adv_logits[:, wrong_voxels].permute(1, 0),
                label[wrong_voxels],
            )
            lmbda = i / (2 * self.num_steps)
            loss = (1 - lmbda) * loss_correct + lmbda * loss_wrong
            if targeted:
                loss = -1 * loss

            # update adversarial image
            grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
            adv_image = adv_image.detach() + self.lr * grad.sign()
            self.delta = adv_image - image
            self._clamp_delta(image)
            adv_image = image + self.delta
            if return_step_by_step:
                adv_imgs.append(
                    self._normalize(adv_image).clone().detach().unsqueeze(0).cpu()
                )

        if return_step_by_step:
            return torch.concatenate(adv_imgs, dim=0)
        else:
            return adv_image.detach()


class DepthPGD(PGD):
    name = "DepthPGD"

    def run(
        self,
        model: DepthEncoderDecoder,
        image: torch.Tensor,
        label: torch.Tensor,
        task_cfg: dict,
        device=None,
        return_step_by_step=False,
    ):
        adv_imgs = []
        image = self._denormalize(image).clone().detach().to(device)  #  [B,C,H,W]
        target = label.clone().detach().to(device)  #  [B,H,W]
        adv_image = image.clone().detach()

        # starting at a uniformly random point
        self.delta = torch.empty_like(adv_image).uniform_(
            -self.eps_budget, self.eps_budget
        )

        self._clamp_delta(image)

        adv_image = adv_image + self.delta

        if return_step_by_step:
            adv_imgs.append(
                self._normalize(adv_image).clone().detach().unsqueeze(0).cpu()
            )

        sig_loss = SigLoss(
            valid_mask=True,
            loss_weight=task_cfg.sig_loss_weight,
            warm_up=True,
            loss_name="loss_depth",
        )
        grad_loss = GradientLoss(
            valid_mask=True,
            loss_weight=task_cfg.grad_loss_weight,
            loss_name="loss_grad",
        )

        for i in range(self.num_steps):
            adv_image.requires_grad = True
            adv_preds = model.whole_inference(
                self._normalize(adv_image), None, rescale=True
            ).squeeze(1)
            loss = sig_loss(adv_preds, target) + grad_loss(adv_preds, target)
            grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
            adv_image = adv_image.detach() + self.lr * grad.sign()
            self.delta = adv_image - image
            self._clamp_delta(image)
            adv_image = image + self.delta
            if return_step_by_step:
                adv_imgs.append(
                    self._normalize(adv_image).clone().detach().unsqueeze(0).cpu()
                )

        if return_step_by_step:
            return torch.concatenate(adv_imgs, dim=0)
        else:
            return adv_image.detach()


pgd_dict: Dict[str, PGD] = {
    "sign": PGD_Sign,
    "downstream": DownstreamPGD,
    "segpgd": SegPGD,
    "autoattack": AutoAttack,
}
