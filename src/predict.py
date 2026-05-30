import argparse
from pathlib import Path

import torch
from PIL import Image

try:
    from .data_utils import build_transforms
    from .model import build_model
except ImportError:
    from data_utils import build_transforms
    from model import build_model


def load_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    architecture = checkpoint.get("architecture", "custom_cnn")
    model = build_model(num_classes=len(class_names), architecture=architecture, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


@torch.no_grad()
def predict_image(model, image_path: str, class_names, image_size: int, device, top_k: int = 3):
    _, eval_transform = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    return [
        {"class_name": class_names[idx.item()], "probability": value.item()}
        for value, idx in zip(values, indices)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict waste class for one image.")
    parser.add_argument("--model_path", default="outputs/best_model.pth")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model, class_names, image_size = load_model(args.model_path, device)
    predictions = predict_image(model, args.image_path, class_names, image_size, device, args.top_k)
    for item in predictions:
        print(f"{item['class_name']}: {item['probability']:.4f}")


if __name__ == "__main__":
    main()
