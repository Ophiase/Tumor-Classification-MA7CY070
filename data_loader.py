
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from common import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, VALID_SPLIT


def limit_dataset(dataset: datasets.ImageFolder, max_images: int) -> datasets.ImageFolder:
    class_counts = {cls: 0 for cls in dataset.classes}
    indices = []
    for idx, (_, label) in enumerate(dataset.samples):
        class_name = dataset.classes[label]
        if class_counts[class_name] < max_images:
            indices.append(idx)
            class_counts[class_name] += 1
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset.targets = [dataset.targets[i] for i in indices]
    return dataset


def create_data_loaders(train_dir: str = TRAIN_DIR,
                        test_dir: str = TEST_DIR,
                        img_size: Tuple[int, int] = IMG_SIZE,
                        batch_size: int = BATCH_SIZE,
                        valid_split: float = VALID_SPLIT,
                        max_images_per_class: int = None
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    full_train = datasets.ImageFolder(train_dir, transform=tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=tfms)

    if max_images_per_class is not None:
        full_train = limit_dataset(full_train, max_images_per_class)
        test_ds = limit_dataset(test_ds, max_images_per_class)

    val_size = int(len(full_train) * valid_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )
