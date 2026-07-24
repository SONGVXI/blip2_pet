"""Model definitions for image-only and future image-text pet classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    """ImageNet-pretrained ResNet-18 that emits a 512-dimensional feature vector."""

    feature_dim = 512

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        """Encode ``(batch_size, 3, height, width)`` images into ``(batch_size, 512)``."""
        return self.backbone(images)


class ImageOnlyClassifier(nn.Module):
    """ResNet-18 image encoder followed by a linear pet-breed classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.classifier = nn.Linear(ImageEncoder.feature_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits with shape ``(batch_size, num_classes)``."""
        image_features = self.image_encoder(images)
        return self.classifier(image_features)


class FusionClassifier(nn.Module):
    """Reserved interface for the later image-text concatenation classifier.

    Text encoding and feature concatenation are deliberately not implemented in
    this baseline stage.  The class exists so later modules can depend on a
    stable import path without introducing a second model API.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_encoder = ImageEncoder()

    def forward(self, images: Tensor, captions: Tensor) -> Tensor:
        raise NotImplementedError(
            "FusionClassifier will be implemented after the text encoder module."
        )

