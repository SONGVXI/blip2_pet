"""使用冻结的 BLIP-2 为 Oxford-IIIT Pet 图片生成 caption。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from dataset import SELECTED_CLASSES, load_pet_datasets


MODEL_NAME = "Salesforce/blip2-opt-2.7b"
DEFAULT_OUTPUT_PATH = Path("./captions/captions.json")
def load_blip2_model(
    model_name: str = MODEL_NAME,
) -> tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.device]:
    """加载 BLIP-2，并将模型设置为只推理模式。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(model_name)
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

    processor, model, device = load_blip2_model()
    # 保留 PIL 图片交给 BLIP-2 processor 处理，避免提前使用 ToTensor。
    datasets = load_pet_datasets(
        root=data_root,
        seed=seed,
        transform=lambda image: image,
    )
    model_dtype = next(model.parameters()).dtype

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

            if max_samples is not None and generated_count >= max_samples:
                break

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return records


def main() -> None:
    """生成 5 条 caption 作为模块测试。"""

    records = generate_captions(max_samples=5)
    print(f"已生成 {len(records)} 条 caption")
    for record in records:
        print(f"{record['label']}: {record['caption']}")
    print(f"结果已保存到: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
