from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(data_dir: str, image_size: int, batch_size: int, num_workers: int):
    data_path = Path(data_dir)
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(data_path / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_path / "val", transform=eval_transform)

    test_path = data_path / "test"
    test_dataset = None
    if test_path.exists():
        test_dataset = datasets.ImageFolder(test_path, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, train_dataset.classes
