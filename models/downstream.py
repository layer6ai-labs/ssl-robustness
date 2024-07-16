import os
from typing import Optional

import torch
from models.ssl_models.ssl import Encoder

from torchvision import transforms


class DownstreamClassifier(torch.nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        dataset_name: str,
        lr: int,
        epochs: int,
        num_classes: int,
        device="cuda",
        is_train: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_name = encoder.name
        self.dataset_name = dataset_name
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.num_classes = num_classes
        self.is_train = is_train


class LinearClassifier(DownstreamClassifier):
    def __init__(
        self,
        encoder: Encoder,
        dataset_name: str,
        lr: int,
        epochs: int,
        num_classes: int,
        device="cuda",
        is_train: bool = False,
    ):
        super().__init__(
            encoder, dataset_name, lr, epochs, num_classes, device, is_train
        )
        self.encoder.eval()
        self.clf = torch.nn.Linear(encoder.feature_dim, num_classes)
        self.clf.to(self.device)

    def get_name(self) -> str:
        return f"tmp/classifiers/linear_{self.encoder_name}_{self.dataset_name}_{self.lr}_{self.epochs}.pt"

    def load_clf(self):
        self.clf.load_state_dict(
            torch.load(
                self.get_name(),
                map_location=torch.device("cpu"),
            )
        )

    def save_clf(self):
        if not os.path.exists("tmp/classifiers"):
            os.mkdir("tmp/classifiers")
        path = self.get_name()
        torch.save(self.clf.state_dict(), path)

    def exits_clf(self):
        return os.path.exists(self.get_name())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_r = self.encoder.get_representation(x)
        y = self.clf(img_r)
        return y

    def clf_params(self):
        return self.clf.parameters()


class SegmenterHead(torch.nn.Module):
    def __init__(
        self,
        num_labels: int,
        encoder: Encoder,
        dataset_name: str,
        epochs: int,
        lr: float,
    ):
        super(SegmenterHead, self).__init__()

        self.in_channels = encoder.feature_dim
        self.width = encoder.image_size // encoder.patch_size
        self.height = encoder.image_size // encoder.patch_size
        self.bn = torch.nn.BatchNorm2d(self.in_channels)
        self.classifier = torch.nn.Conv2d(encoder.feature_dim, num_labels, (1, 1))
        self.output_size = [encoder.image_size] * 2
        self.encoder_name = encoder.name
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.lr = lr
        self.num_labels = num_labels

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.bn(embeddings)
        logits = self.classifier(embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=self.output_size, mode="bilinear", align_corners=False
        )

        return logits

    def get_name(self) -> str:
        return f"tmp/segmenters/linear_{self.encoder_name}_{self.dataset_name}_{self.lr}_{self.epochs}.pt"

    def load_segmenter(self):
        self.load_state_dict(
            torch.load(
                self.get_name(),
                map_location=torch.device("cpu"),
            )
        )

    def exists_segmenter(self):
        return os.path.exists(self.get_name())

    def save_segmenter(self):
        os.makedirs("tmp/segmenters", exist_ok=True)
        path = self.get_name()
        torch.save(self.state_dict(), path)


class Segmenter(torch.nn.Module):
    def __init__(
        self, num_labels: int, encoder: Encoder, head: Optional[SegmenterHead] = None
    ):
        super(Segmenter, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        if head is not None:
            self.head = head
        else:
            self.head = SegmenterHead(num_labels=num_labels, encoder=encoder)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder.get_patch_embeddings(images)
        logits = self.head(embeddings)
        return logits


class NormalizationWrapper(torch.nn.Module):
    """
    Wrapper around a model that applies a given transformation
    to the input before passing it to the model.

    Use-case: AutoAttack can't let us normalize the input before,
    so it has to happen inside the model.
    """

    def __init__(self, model: torch.nn.Module, transform: transforms.Compose):
        super().__init__()
        self.model = model
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        return self.model(x)
