# adapted from https://github.com/pantheon5100/DeACL/blob/dc0807e0b2b133fec3c9a3ec2dca6f3a2527cb5e
# /solo/utils/pretrain_dataloader_AdvTraining.py

import random
from typing import Any, Callable, List, Sequence, Optional

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

import albumentations as A

mean_dict = {
    "imagenet": (0.485, 0.456, 0.406),
    "no_norm": (0.0, 0.0, 0.0),
}
std_dict = {
    "imagenet": (0.228, 0.224, 0.225),
    "no_norm": (1.0, 1.0, 1.0),
}


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transform:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transform])


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)


class SegmentationTransform(BaseTransform):
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        crop_size: int = 518,
        mean: Sequence[float] = (123.675 / 255, 116.280 / 255, 103.530 / 255),
        std: Sequence[float] = (58.395 / 255, 57.120 / 255, 57.375 / 255),
        augment: bool = True,
    ):
        # based on DINOv2 config and Perplexity.ai's explanations
        # of the mmseg documentation
        if augment:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=crop_size),
                    A.RandomCrop(crop_size, crop_size),
                    A.Flip(p=0.5),
                    A.Normalize(mean=mean, std=std),
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(
                        max_size=crop_size, interpolation=0
                    ),  # cv2.INTER_NEAREST
                    A.Normalize(mean=mean, std=std),
                ]
            )


class DepthEstimationTransform(BaseTransform):
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        crop_size: Sequence[int] = (480, 640),
        mean: Sequence[float] = (123.675 / 255, 116.280 / 255, 103.530 / 255),
        std: Sequence[float] = (58.395 / 255, 57.120 / 255, 57.375 / 255),
        augment: bool = False,
    ):
        if augment:
            self.transform = A.Compose(
                [
                    A.Resize(height=480, width=640),
                    A.Rotate(p=0.5, limit=2.5),
                    A.Flip(p=0.5),
                    A.RandomCrop(crop_size=(416, 544)),
                    # dict( # Keeping these comments just in case for future reference
                    #     type='ColorAug',
                    #     prob=0.5,
                    #     gamma_range=[0.9, 1.1],
                    #     brightness_range=[0.75, 1.25],
                    #     color_range=[0.9, 1.1]),
                    # eigen_crop=True, -> for both train and valid
                    A.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=480, width=640),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=mean, std=std),
                ]
            )


class ClassificationTranform(BaseTransform):
    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
        augment: bool = True,
        crop_size: int = 224,
    ):
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(crop_size, antialias=True),
                    transforms.RandomCrop(crop_size, padding=4 * crop_size // 32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(crop_size, antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )


def prepare_transform(
    dataset: str,
    augment: bool,
    crop_size: int,
    split: str,
    normalize: Optional[bool] = True,
) -> Any:
    mean = mean_dict["imagenet" if normalize else "no_norm"]
    std = std_dict["imagenet" if normalize else "no_norm"]

    if dataset in ("foodseg103", "ade20k", "cityscapes", "pascal_voc_2012"):
        return SegmentationTransform(
            mean=mean,
            std=std,
            augment=(split == "train") or augment,
            crop_size=crop_size,
        )
    if dataset in ("nyu_depth_v2"):
        return DepthEstimationTransform(
            mean=mean,
            std=std,
            augment=augment,
            crop_size=crop_size,
        )
    return ClassificationTranform(
        mean=mean,
        std=std,
        augment=(split == "train") or augment,
        crop_size=crop_size,
    )


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)
