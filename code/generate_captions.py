"""Generate frozen BLIP-2 captions for the Oxford-IIIT Pet samples.

The dataset module is the single source of samples.  It should expose either:

* ``get_caption_samples(split)`` returning dictionaries with ``image_path`` and
  ``label`` keys; or
* ``PetDataset(split=...)`` whose ``caption_samples`` attribute uses that same
  dictionary format.

Example:
    python code/generate_captions.py --split trainval --limit 5
    python code/generate_captions.py --split all --output captions/captions.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "captions" / "captions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate captions with frozen BLIP-2.")
    parser.add_argument("--split", default="all", help="Dataset split supplied to dataset.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the JSON caption file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to caption (use 5 for a quick test).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=30)
    return parser.parse_args()


def load_caption_samples(split: str) -> list[dict[str, Any]]:
    """Load image paths and labels from ``code/dataset.py`` without changing it."""
    code_directory = str(Path(__file__).resolve().parent)
    if code_directory not in sys.path:
        sys.path.insert(0, code_directory)
    dataset_module = importlib.import_module("dataset")

    if callable(getattr(dataset_module, "get_caption_samples", None)):
        samples: Iterable[dict[str, Any]] = dataset_module.get_caption_samples(split=split)
    elif hasattr(dataset_module, "PetDataset"):
        dataset = dataset_module.PetDataset(split=split)
        samples = getattr(dataset, "caption_samples", None)
        if samples is None:
            raise RuntimeError(
                "dataset.PetDataset must expose a caption_samples attribute containing "
                "{'image_path': ..., 'label': ...} records."
            )
    else:
        raise RuntimeError(
            "dataset.py has no caption sample interface. Implement get_caption_samples(split) "
            "in the dataset module before generating captions."
        )

    records = []
    for sample in samples:
        if not isinstance(sample, dict) or {"image_path", "label"} - sample.keys():
            raise ValueError(
                "Each dataset sample must be a dictionary with 'image_path' and 'label' keys."
            )
        records.append({"image_path": str(sample["image_path"]), "label": str(sample["label"])})
    return records


def load_blip2(model_name: str = DEFAULT_MODEL_NAME) -> tuple[Blip2Processor, Blip2ForConditionalGeneration, torch.device]:
    """Load BLIP-2 in inference mode only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, device


def generate_caption(
    image_path: str,
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Run a single frozen BLIP-2 inference pass."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

    # Caption generation is inference-only: no gradients or parameter updates.
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def generate_captions(samples: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, str]]:
    """Generate one caption record for every supplied sample."""
    processor, model, device = load_blip2()
    captions: list[dict[str, str]] = []
    for sample in tqdm(samples, desc="Generating BLIP-2 captions"):
        captions.append(
            {
                "image_path": sample["image_path"],
                "label": sample["label"],
                "caption": generate_caption(
                    sample["image_path"], processor, model, device, max_new_tokens
                ),
            }
        )
    return captions


def save_captions(captions: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(captions, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    samples = load_caption_samples(args.split)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No samples were returned by dataset.py.")

    captions = generate_captions(samples, args.max_new_tokens)
    save_captions(captions, args.output)
    print(f"Saved {len(captions)} captions to {args.output}")
    for record in captions[:5]:
        print(f"{record['label']}: {record['caption']}")


if __name__ == "__main__":
    # The default entry point is intentionally a five-image smoke test.
    if len(sys.argv) == 1:
        sys.argv.extend(["--limit", "5"])
    main()
