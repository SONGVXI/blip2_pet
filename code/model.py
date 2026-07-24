"""BLIP-2 宠物品种识别项目的图像基线模型定义。"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from text_encoder import TextEncoder


NUM_CLASSES = 10
IMAGE_FEATURE_DIM = 512


class ImageEncoder(nn.Module):
    """使用 ImageNet 预训练 ResNet-18 提取 512 维图像特征。"""

    output_dim = IMAGE_FEATURE_DIM

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        if pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except (TypeError, NameError):
                # 兼容较旧的 torchvision 版本。
                backbone = resnet18(pretrained=True)
        else:
            try:
                backbone = resnet18(weights=None)
            except TypeError:
                backbone = resnet18(pretrained=False)

        # ResNet-18 的 fc 输入是 512 维；替换为 Identity 后直接输出该特征。
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        """输入 [batch, 3, height, width]，输出 [batch, 512]。"""

        return self.backbone(images)


class ImageOnlyClassifier(nn.Module):
    """Image-only baseline：ResNet-18 特征接一个 10 类线性分类层。"""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.classifier = nn.Linear(self.image_encoder.output_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """输入 [batch, 3, height, width]，输出分类 logits [batch, num_classes]。"""

        image_features = self.image_encoder(images)
        return self.classifier(image_features)


class FusionClassifier(nn.Module):
    """图像-文本融合接口。

    文本编码器暂不在本模块实现。调用者需要传入已经提取好的
    ``text_features``，其形状应为 [batch, text_feature_dim]。
    """

    def __init__(
        self,
        text_feature_dim: int,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.text_feature_dim = text_feature_dim
        self.classifier = nn.Linear(
            self.image_encoder.output_dim + text_feature_dim,
            num_classes,
        )

    def forward(
        self,
        images: Tensor,
        text_features: Optional[Tensor] = None,
    ) -> Tensor:
        """拼接图像特征和外部文本特征，输出分类 logits。"""

        if text_features is None:
            raise ValueError(
                "FusionClassifier 暂不包含文本编码器，请传入 text_features。"
            )
        if text_features.ndim != 2:
            raise ValueError("text_features 的形状必须是 [batch, text_feature_dim]。")
        if text_features.shape[0] != images.shape[0]:
            raise ValueError("images 和 text_features 的 batch size 必须一致。")
        if text_features.shape[1] != self.text_feature_dim:
            raise ValueError(
                f"text_features 的维度应为 {self.text_feature_dim}，"
                f"实际为 {text_features.shape[1]}。"
            )

        image_features = self.image_encoder(images)
        fused_features = torch.cat((image_features, text_features), dim=1)
        return self.classifier(fused_features)


class ImageTextFusionClassifier(nn.Module):
    """ResNet-18 图像特征与 GRU 文本特征的多模态分类器。"""

    def __init__(
        self,
        text_encoder: TextEncoder,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.text_encoder = text_encoder

        fusion_dim = self.image_encoder.output_dim + text_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        images: Tensor,
        caption_tokens: Tensor | list[str],
    ) -> Tensor:
        """输入图像和 caption tokens，输出 [batch_size, num_classes]。"""

        image_features = self.image_encoder(images)
        text_features = self.text_encoder(caption_tokens)

        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError("image 和 caption_tokens 的 batch size 必须一致。")

        fused_features = torch.cat((image_features, text_features), dim=1)
        return self.classifier(fused_features)
