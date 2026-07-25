"""完整测试脚本：评估模型并绘制训练曲线。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import evaluate


def read_log(log_path: Path) -> List[Dict[str, float]]:
    """读取训练日志中的 epoch、loss 和 validation accuracy。"""

    if not log_path.exists():
        raise FileNotFoundError(f"未找到训练日志：{log_path}")

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records: List[Dict[str, float]] = []
    for row in rows:
        loss_key = "fusion_train_loss" if "fusion_train_loss" in row else "train_loss"
        records.append(
            {
                "epoch": float(row["epoch"]),
                "loss": float(row[loss_key]),
                "accuracy": float(row["validation_accuracy"]),
            }
        )
    return records


def save_curve(
    records: List[Dict[str, float]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    """绘制并保存一条训练曲线。"""

    epochs = [record["epoch"] for record in records]
    values = [record[value_key] for record in records]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, values, marker="o", color=color, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(project_root: Path, results_dir: Path) -> List[Path]:
    """读取两个模型的日志并保存 loss、accuracy 曲线。"""

    image_only_records = read_log(project_root / "logs" / "image_only_train.csv")
    fusion_records = read_log(project_root / "logs" / "fusion_train.csv")

    output_paths = [
        results_dir / "image_only_loss.png",
        results_dir / "fusion_loss.png",
        results_dir / "image_only_accuracy.png",
        results_dir / "fusion_accuracy.png",
    ]
    save_curve(
        image_only_records,
        "loss",
        "Image-only Training Loss",
        "Loss",
        output_paths[0],
        "tab:blue",
    )
    save_curve(
        fusion_records,
        "loss",
        "Image-text Fusion Training Loss",
        "Loss",
        output_paths[1],
        "tab:orange",
    )
    save_curve(
        image_only_records,
        "accuracy",
        "Image-only Validation Accuracy",
        "Accuracy",
        output_paths[2],
        "tab:blue",
    )
    save_curve(
        fusion_records,
        "accuracy",
        "Image-text Fusion Validation Accuracy",
        "Accuracy",
        output_paths[3],
        "tab:orange",
    )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate both models and save training curves."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluation_results = evaluate(
        image_only_path=project_root / "checkpoints" / "image_only_best.pth",
        fusion_path=project_root / "checkpoints" / "fusion_best.pth",
        data_root=project_root / "data",
        captions_path=project_root / "captions" / "captions.json",
        output_path=results_dir / "predictions.json",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("\n评估摘要:")
    print(f"类别数量: {evaluation_results['num_classes']}")
    print(f"图片数量: {evaluation_results['num_images']}")
    print(
        f"image-only test accuracy: "
        f"{evaluation_results['image_only_test_accuracy']:.4f}"
    )
    print(
        f"fusion test accuracy: "
        f"{evaluation_results['fusion_test_accuracy']:.4f}"
    )

    print("\n随机预测结果:")
    for index, sample in enumerate(evaluation_results["random_samples"], start=1):
        print(f"[{index}]")
        print(f"图片路径: {sample['image_path']}")
        print(f"真实类别: {sample['true_label']}")
        print(f"caption: {sample['caption']}")
        print(f"image-only预测: {sample['image_only_prediction']}")
        print(f"fusion预测: {sample['fusion_prediction']}")

    curve_paths = plot_training_curves(project_root, results_dir)
    print("\n曲线已保存:")
    for curve_path in curve_paths:
        print(curve_path)
    print(f"预测结果已保存: {results_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
