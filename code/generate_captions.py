"""使用冻结的 BLIP-2 为 Oxford-IIIT Pet 图片生成 caption。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from dataset import SELECTED_CLASSES, load_pet_datasets


MODEL_NAME = "Salesforce/blip2-opt-2.7b"
DEFAULT_OUTPUT_PATH = Path("./captions/captions.json")
def load_blip2_model(
    model_name: str = MODEL_NAME,
) -> tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.device]:
    """加载 BLIP-2，并将模型设置为只推理模式。

    8GB 显存设备使用 Accelerate 自动分配：尽量把模型放在 GPU，
    显存不足的部分自动 offload 到 CPU，避免直接执行完整的 ``model.to(cuda)``。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(model_name)
    if device.type == "cuda":
        offload_dir = Path("checkpoints/blip2_offload")
        offload_dir.mkdir(parents=True, exist_ok=True)
        try:
            import bitsandbytes  # noqa: F401

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "7600MiB", "cpu": "32GiB"},
                offload_folder=str(offload_dir),
                offload_state_dict=True,
            )
        except ImportError as error:
            raise RuntimeError(
                "4-bit 加载需要 accelerate 和 bitsandbytes，请运行："
                "pip install -U accelerate bitsandbytes"
            ) from error
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        model.to(device)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, device


def _move_inputs_to_device(
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """移动输入；仅将浮点输入转换为模型精度，避免改变 token 类型。"""

    moved_inputs: Dict[str, torch.Tensor] = {}
    for name, value in inputs.items():
        if value.is_floating_point():
            moved_inputs[name] = value.to(device=device, dtype=model_dtype)
        else:
            moved_inputs[name] = value.to(device)
    return moved_inputs


def generate_captions(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_samples: Optional[int] = None,
    data_root: str = "data",
    seed: int = 42,
) -> List[Dict[str, str]]:
    """为数据集图片生成 caption 并保存为 JSON。

    ``max_samples=None`` 时处理全部 450 张图片；传入整数可限制生成数量，
    便于先进行小规模测试。
    """

    print("[1/4] 正在加载 BLIP-2 模型...", flush=True)
    processor, model, device = load_blip2_model()
    print(f"[2/4] 模型加载完成，运行设备: {device}", flush=True)
    # 保留 PIL 图片交给 BLIP-2 processor 处理，避免提前使用 ToTensor。
    print("[3/4] 正在读取 Oxford-IIIT Pet 数据集...", flush=True)
    datasets = load_pet_datasets(
        root=data_root,
        seed=seed,
        transform=lambda image: image,
    )
    total_available = sum(len(dataset) for dataset in datasets.values())
    print(f"数据集读取完成，共 {total_available} 张图片。", flush=True)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    records: List[Dict[str, str]] = []
    generated_count = 0
    with torch.no_grad():
        for split_name in ("train", "validation", "test"):
            dataset = datasets[split_name]
            for index in range(len(dataset)):
                if max_samples is not None and generated_count >= max_samples:
                    break

                sample = dataset[index]
                inputs = processor(
                    images=sample["image"],
                    return_tensors="pt",
                )
                inputs = _move_inputs_to_device(inputs, device, model_dtype)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )
                # 不传入文本 prompt，因此这里直接解码 BLIP-2 生成的 caption。
                caption = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip()

                label = int(sample["label"])
                records.append(
                    {
                        "image_path": str(sample["image_path"]),
                        "label": SELECTED_CLASSES[label],
                        "caption": caption,
                    }
                )
                generated_count += 1
                if generated_count == 1 or generated_count % 10 == 0:
                    print(
                        f"已生成 {generated_count}/{max_samples or total_available} 条 caption",
                        flush=True,
                    )

            if max_samples is not None and generated_count >= max_samples:
                break

    print("[4/4] 正在保存 captions.json...", flush=True)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print(f"caption 生成完成，共 {len(records)} 条。", flush=True)
    return records


def main() -> None:
    """默认生成 5 条测试 caption，使用 --all 时生成全部 caption。"""

    parser = argparse.ArgumentParser(
        description="Generate captions with the frozen BLIP-2 model."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="为全部数据集图片生成 caption。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制生成数量；未指定 --all 时默认生成 5 条。",
    )
    args = parser.parse_args()

    max_samples = None if args.all else (args.max_samples or 5)
    records = generate_captions(max_samples=None)
    print(f"已生成 {len(records)} 条 caption")
    for record in records:
        print(f"{record['label']}: {record['caption']}")
    if len(records) > 5:
        print(f"其余 {len(records) - 5} 条 caption 已写入文件。")
    print(f"结果已保存到: {DEFAULT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
