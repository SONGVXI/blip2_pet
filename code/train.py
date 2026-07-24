"""Train the ResNet-18 image-only pet-breed baseline."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from tqdm import tqdm

from model import ImageOnlyClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "image_only_best.pth"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "image_only_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image-only ResNet-18 baseline.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (5 to 10).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    return parser.parse_args()


def get_dataloaders(args: argparse.Namespace) -> tuple[Any, Any]:
    """Create image-only loaders through the project's dataset module."""
    code_directory = str(Path(__file__).resolve().parent)
    if code_directory not in sys.path:
        sys.path.insert(0, code_directory)
    dataset_module = importlib.import_module("dataset")
    get_dataloader = getattr(dataset_module, "get_dataloader", None)
    if not callable(get_dataloader):
        raise RuntimeError(
            "dataset.py must implement get_dataloader(split, batch_size, shuffle) "
            "before image-only training can start."
        )

    train_loader = get_dataloader(
        split=args.train_split, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(split=args.val_split, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def unpack_image_batch(batch: Any) -> tuple[Tensor, Tensor]:
    """Extract images and integer labels from common project DataLoader formats."""
    if isinstance(batch, dict):
        images = batch.get("images", batch.get("image"))
        labels = batch.get("labels", batch.get("label"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[-1]
    else:
        raise TypeError("A batch must be a dict or tuple containing images and labels.")

    if not isinstance(images, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("Dataset batches must contain torch Tensor images and labels.")
    return images, labels.long()


def train_one_epoch(
    model: ImageOnlyClassifier,
    loader: Any,
    optimizer: Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        images, labels = unpack_image_batch(batch)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The training DataLoader is empty.")
    return total_loss / total_samples


def validate(
    model: ImageOnlyClassifier,
    loader: Any,
    device: torch.device,
) -> float:
    """Evaluate classification accuracy without tracking gradients."""
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images, labels = unpack_image_batch(batch)
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("The validation DataLoader is empty.")
    return correct / total_samples


def write_log_row(log_file: Path, epoch: int, train_loss: float, val_accuracy: float) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "validation_accuracy"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_accuracy:.6f}"])


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: ImageOnlyClassifier,
    optimizer: Adam,
    val_accuracy: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_accuracy,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    if not 5 <= args.epochs <= 10:
        raise ValueError("--epochs must be between 5 and 10.")
    if args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("--batch-size and --lr must be positive.")

    train_loader, val_loader = get_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageOnlyClassifier(num_classes=args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy = validate(model, val_loader, device)
        write_log_row(args.log_file, epoch, train_loss, val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(args.checkpoint, epoch, model, optimizer, val_accuracy)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_accuracy={val_accuracy:.4f}"
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
