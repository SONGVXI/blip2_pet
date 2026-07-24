from torchvision.datasets import OxfordIIITPet

dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True
)

print(len(dataset.classes))

for i in range(5):
    image, label = dataset[i]

    print(
        i,
        label,
        dataset.classes[label]
    )
SELECTED_CLASSES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Persian",
    "Siamese",
    "american_bulldog",
    "american_pit_bull_terrier",
    "english_cocker_spaniel",
    "english_setter",
    "staffordshire_bull_terrier"
]
print(dataset._images[0])