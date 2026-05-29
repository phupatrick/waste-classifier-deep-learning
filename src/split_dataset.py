import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split an image dataset into train/val/test folders.")
    parser.add_argument("--input_dir", required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def copy_images(files, class_name, split_name, output_dir):
    target_dir = Path(output_dir) / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    class_dirs = [path for path in input_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError("input_dir must contain class subfolders.")

    for class_dir in class_dirs:
        images = [
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(images)

        train_end = int(len(images) * args.train_ratio)
        val_end = train_end + int(len(images) * args.val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        for split_name, files in splits.items():
            copy_images(files, class_dir.name, split_name, args.output_dir)

        print(
            f"{class_dir.name}: "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()
