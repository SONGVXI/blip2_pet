"""Image-only 宠物品种分类模型训练脚本。"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import load_pet_datasets
from model import ImageOnlyClassifier


DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
NUM_CLASSES = 10


def set_seed(seed: int) -> None:
    """设置随机种子，便于复现实验结果。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transform() -> transforms.Compose:
    """构造适配 ResNet-18 的图像预处理。"""

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """执行一个训练 epoch，返回平均训练损失。"""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """执行验证并返回准确率。"""

    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device, dtype=torch.long)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    data_root: str | Path = "data",
    checkpoint_path: str | Path = "checkpoints/image_only_best.pth",
    log_path: str | Path = "logs/image_only_train.csv",
    seed: int = 42,
) -> Tuple[Path, Path]:
    """训练 image-only 模型并保存最佳 checkpoint 和训练日志。"""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform = build_image_transform()
    datasets = load_pet_datasets(
        root=str(data_root),
        seed=seed,
        transform=image_transform,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ImageOnlyClassifier(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    best_validation_accuracy = -1.0
    with log_file.open("w", newline="", encoding="utf-8") as file:
        logger = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "validation_accuracy"],
        )
        logger.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_accuracy = validate(model, validation_loader, device)
            logger.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "validation_accuracy": f"{validation_accuracy:.6f}",
                }
            )
            file.flush()

            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"train loss: {train_loss:.4f} | "
                f"validation accuracy: {validation_accuracy:.4f}"
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "validation_accuracy": validation_accuracy,
                        "num_classes": NUM_CLASSES,
                    },
                    checkpoint_file,
                )

    print(f"Best checkpoint: {checkpoint_file}")
    print(f"Training log: {log_file}")
    return checkpoint_file, log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image-only baseline.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_root=project_root / "data",
        checkpoint_path=project_root / "checkpoints" / "image_only_best.pth",
        log_path=project_root / "logs" / "image_only_train.csv",
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
