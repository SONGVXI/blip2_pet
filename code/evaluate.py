"""评估 image-only 与 image-text fusion 模型。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import SELECTED_CLASSES, load_pet_datasets
from model import ImageOnlyClassifier, ImageTextFusionClassifier
from text_encoder import TextEncoder
from train import CaptionDataset, build_image_transform, load_captions


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": checkpoint}


def _build_models(
    image_only_path: Path,
    fusion_path: Path,
    device: torch.device,
) -> Tuple[ImageOnlyClassifier, ImageTextFusionClassifier]:
    image_checkpoint = _load_checkpoint(image_only_path, device)
    fusion_checkpoint = _load_checkpoint(fusion_path, device)

    image_only_model = ImageOnlyClassifier(
        num_classes=len(SELECTED_CLASSES),
        pretrained=False,
    )
    image_only_model.load_state_dict(image_checkpoint["model_state_dict"])
    image_only_model.to(device).eval()

    text_vocab = fusion_checkpoint.get("text_vocab")
    if not isinstance(text_vocab, dict):
        raise KeyError("fusion checkpoint 中缺少 text_vocab，无法重建 TextEncoder。")
    text_encoder = TextEncoder(vocab=text_vocab, hidden_dim=128)
    fusion_model = ImageTextFusionClassifier(
        text_encoder=text_encoder,
        num_classes=len(SELECTED_CLASSES),
        pretrained=False,
    )
    fusion_model.load_state_dict(fusion_checkpoint["model_state_dict"])
    fusion_model.to(device).eval()
    return image_only_model, fusion_model


@torch.no_grad()
def evaluate(
    image_only_path: str | Path = "checkpoints/image_only_best.pth",
    fusion_path: str | Path = "checkpoints/fusion_best.pth",
    data_root: str | Path = "data",
    captions_path: str | Path = "captions/captions.json",
    output_path: str | Path = "results/predictions.json",
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, object]:
    """在 test 集上评估两个模型并保存预测结果。"""

    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]

    captions_by_path, _ = load_captions(captions_path, project_root)
    image_datasets = load_pet_datasets(
        root=str(data_root),
        seed=seed,
        transform=build_image_transform(),
    )
    test_dataset = CaptionDataset(
        image_datasets["test"],
        captions_by_path,
        project_root,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    image_only_model, fusion_model = _build_models(
        Path(image_only_path),
        Path(fusion_path),
        device,
    )

    prediction_records: List[Dict[str, object]] = []
    image_only_correct = 0
    fusion_correct = 0
    total = 0

    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device, dtype=torch.long)
        image_only_logits = image_only_model(images)
        fusion_logits = fusion_model(images, batch["caption"])
        image_only_predictions = image_only_logits.argmax(dim=1)
        fusion_predictions = fusion_logits.argmax(dim=1)

        image_only_correct += (image_only_predictions == labels).sum().item()
        fusion_correct += (fusion_predictions == labels).sum().item()

        for index in range(labels.size(0)):
            true_label = int(labels[index].item())
            image_only_label = int(image_only_predictions[index].item())
            fusion_label = int(fusion_predictions[index].item())
            prediction_records.append(
                {
                    "image_path": str(batch["image_path"][index]),
                    "true_label": SELECTED_CLASSES[true_label],
                    "caption": str(batch["caption"][index]),
                    "image_only_prediction": SELECTED_CLASSES[image_only_label],
                    "fusion_prediction": SELECTED_CLASSES[fusion_label],
                }
            )
        total += labels.size(0)

    image_only_accuracy = image_only_correct / max(total, 1)
    fusion_accuracy = fusion_correct / max(total, 1)
    sample_count = min(3, len(prediction_records))
    random_samples = random.sample(prediction_records, sample_count)
    results: Dict[str, object] = {
        "num_classes": len(SELECTED_CLASSES),
        "num_images": total,
        "image_only_test_accuracy": image_only_accuracy,
        "fusion_test_accuracy": fusion_accuracy,
        "random_samples": random_samples,
        "predictions": prediction_records,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"类别数量: {len(SELECTED_CLASSES)}")
    print(f"图片数量: {total}")
    print(f"image-only test accuracy: {image_only_accuracy:.4f}")
    print(f"fusion test accuracy: {fusion_accuracy:.4f}")
    print("随机预测结果:")
    for sample in random_samples:
        print(
            f"  {sample['image_path']} | "
            f"真实: {sample['true_label']} | "
            f"image-only: {sample['image_only_prediction']} | "
            f"fusion: {sample['fusion_prediction']}"
        )
    print(f"结果已保存到: {output_file}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate both classification models.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    evaluate(
        image_only_path=project_root / "checkpoints" / "image_only_best.pth",
        fusion_path=project_root / "checkpoints" / "fusion_best.pth",
        data_root=project_root / "data",
        captions_path=project_root / "captions" / "captions.json",
        output_path=project_root / "results" / "predictions.json",
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
