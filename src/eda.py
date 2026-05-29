import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create simple EDA tables and charts for the waste dataset.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/eda")
    return parser.parse_args()


def scan_dataset(data_dir: str):
    rows = []
    for split_dir in Path(data_dir).iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                except OSError:
                    width, height = None, None
                rows.append(
                    {
                        "split": split_dir.name,
                        "class_name": class_dir.name,
                        "file_path": str(image_path),
                        "width": width,
                        "height": height,
                    }
                )
    return pd.DataFrame(rows)


def plot_class_distribution(summary_df, output_path):
    pivot = summary_df.pivot(index="class_name", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Số lượng ảnh theo lớp")
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Số ảnh")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(args.data_dir)
    if df.empty:
        raise ValueError("No images found. Please check data_dir.")

    summary_df = (
        df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "class_name"])
    )

    df.to_csv(output_dir / "dataset_images.csv", index=False)
    summary_df.to_csv(output_dir / "class_distribution.csv", index=False)
    plot_class_distribution(summary_df, output_dir / "class_distribution.png")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
