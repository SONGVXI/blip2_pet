"""Oxford-IIIT Pet 数据读取与小规模划分。

本模块只负责数据读取，不包含模型或训练代码。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


SELECTED_CLASSES: Tuple[str, ...] = (
    "Abyssinian",
    "Bengal",
    "Birman",
    "Persian",
    "Siamese",
    "american_bulldog",
    "american_pit_bull_terrier",
    "english_cocker_spaniel",
    "english_setter",
    "staffordshire_bull_terrier",
)

TRAIN_PER_CLASS = 30
VALIDATION_PER_CLASS = 5
TEST_PER_CLASS = 10


class PetDataset(Dataset):
    """带有统一标签和图片路径返回格式的 Oxford-IIIT Pet 子集。"""

    def __init__(
        self,
        source_dataset: OxfordIIITPet,
        records: Sequence[Tuple[int, int, str]],
        class_names: Sequence[str],
    ) -> None:
        self.source_dataset = source_dataset
        self.records = list(records)
        self.classes = list(class_names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        source_index, label, image_path = self.records[index]
        image, _ = self.source_dataset[source_index]
        return {
            "image": image,
            "label": label,
            "image_path": image_path,
        }


def _get_image_paths(dataset: OxfordIIITPet) -> List[str]:
    """读取 torchvision 数据集保存的图片路径，兼容常见 torchvision 版本。"""

    image_paths = getattr(dataset, "_images", None)
    if image_paths is None:
        image_paths = getattr(dataset, "_image_files", None)
    if image_paths is None:
        raise RuntimeError("当前 torchvision 版本无法读取 Oxford-IIIT Pet 图片路径。")
    return [str(path) for path in image_paths]


def _resolve_source_class_name(
    class_to_idx: Dict[str, int], requested_name: str
) -> str:
    """兼容 torchvision 中类别名的大小写、空格和下划线差异。"""

    def normalise(name: str) -> str:
        return "".join(character.lower() for character in name if character.isalnum())

    normalised_names = {
        normalise(source_name): source_name for source_name in class_to_idx
    }
    try:
        return normalised_names[normalise(requested_name)]
    except KeyError as error:
        raise KeyError(f"数据集中找不到类别：{requested_name}") from error


def _build_records(
    dataset: OxfordIIITPet,
    class_to_label: Dict[str, int],
    samples_per_class: int,
    seed: int,
) -> List[Tuple[int, int, str]]:
    """从一个官方 split 中按类别均衡、可复现地抽样。"""

    labels = list(getattr(dataset, "_labels"))
    image_paths = _get_image_paths(dataset)
    source_class_to_idx = dataset.class_to_idx
    rng = random.Random(seed)
    records: List[Tuple[int, int, str]] = []

    for class_name, label in class_to_label.items():
        source_name = _resolve_source_class_name(source_class_to_idx, class_name)
        source_label = source_class_to_idx[source_name]
        candidates = [
            index for index, source_target in enumerate(labels)
            if source_target == source_label
        ]
        rng.shuffle(candidates)
        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"类别 {class_name} 只有 {len(candidates)} 张图片，"
                f"无法抽取 {samples_per_class} 张。"
            )
        records.extend(
            (index, label, image_paths[index])
            for index in candidates[:samples_per_class]
        )

    return records


def load_pet_datasets(
    root: str = "./data",
    download: bool = True,
    seed: int = 42,
    transform: Optional[object] = None,
) -> Dict[str, PetDataset]:
    """下载并创建 train、validation、test 三个均衡数据集。

    train/validation 来自官方 trainval split，test 来自官方 test split，
    因此 train/test 不会出现同一张图片。默认总数为 300/50/100。
    """

    if transform is None:
        transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor()])
        print("正在运行图像增强变换")

    data_root = Path(root)
    print(str(data_root))
    trainval_source = OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        transform=transform,
        download=download,
    )
    test_source = OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="category",
        transform=transform,
        download=download,
    )
    
    class_to_label = {name: label for label, name in enumerate(SELECTED_CLASSES)}
    trainval_records = _build_records(
        trainval_source, class_to_label, TRAIN_PER_CLASS + VALIDATION_PER_CLASS, seed
    )
    test_records = _build_records(
        test_source, class_to_label, TEST_PER_CLASS, seed + 1
    )

    trainval_paths = {record[2] for record in trainval_records}
    test_paths = {record[2] for record in test_records}
    overlap = trainval_paths.intersection(test_paths)
    if overlap:
        raise RuntimeError(f"检测到 train/test 图片重复：{len(overlap)} 张")

    # _build_records 按类别追加记录；在每个类别内部切分，保证每个 split
    # 都包含所有类别，同时 train 和 validation 仍来自不重复的图片。
    train_records: List[Tuple[int, int, str]] = []
    validation_records: List[Tuple[int, int, str]] = []
    for label in range(len(SELECTED_CLASSES)):
        class_records = [record for record in trainval_records if record[1] == label]
        train_records.extend(class_records[:TRAIN_PER_CLASS])
        validation_records.extend(class_records[TRAIN_PER_CLASS:])

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(validation_records)

    return {
        "train": PetDataset(trainval_source, train_records, SELECTED_CLASSES),
        "validation": PetDataset(
            trainval_source, validation_records, SELECTED_CLASSES
        ),
        "test": PetDataset(test_source, test_records, SELECTED_CLASSES),
    }


def main() -> None:
    """下载数据并打印数据集概况及一个样本。"""

    datasets = load_pet_datasets()
    total_count = sum(len(dataset) for dataset in datasets.values())
    print(f"图片数量: {total_count}")
    print(f"类别数量: {len(SELECTED_CLASSES)}")
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {len(dataset)}")

    sample = datasets["train"][0]
    print("sample:")
    print(f"  image type: {type(sample['image']).__name__}")
    print(f"  label: {sample['label']} ({SELECTED_CLASSES[sample['label']]})")
    print(f"  image_path: {sample['image_path']}")


if __name__ == "__main__":
    main()
