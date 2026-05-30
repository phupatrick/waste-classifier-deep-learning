import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

try:
    from .data_utils import build_dataloaders
    from .metrics import classification_metrics
    from .model import build_model
    from .plots import plot_confusion_matrix, plot_training_history
except ImportError:
    from data_utils import build_dataloaders
    from metrics import classification_metrics
    from model import build_model
    from plots import plot_confusion_matrix, plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for household waste classification.")
    parser.add_argument("--data_dir", default="data", help="Folder containing train/val/test subfolders.")
    parser.add_argument("--output_dir", default="outputs", help="Folder to save models and reports.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=["custom_cnn", "mobilenet_v2"], default="custom_cnn")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights when available.")
    parser.add_argument("--class_weights", action="store_true", help="Balance loss for uneven class counts.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, dataloader, criterion, optimizer, device, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size, args.num_workers
    )

    model = build_model(
        num_classes=len(class_names),
        architecture=args.architecture,
        pretrained=args.pretrained,
    ).to(device)

    if args.class_weights:
        targets = np.array(train_loader.dataset.targets)
        counts = np.bincount(targets, minlength=len(class_names))
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_acc = 0.0
    history = []
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "val_acc": best_val_acc,
                    "architecture": args.architecture,
                    "pretrained": args.pretrained,
                },
                best_model_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(history_df, output_dir / "training_history.png")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    y_true, y_pred = collect_predictions(model, eval_loader, device)
    results = classification_metrics(y_true, y_pred, class_names)

    results["report_df"].to_csv(output_dir / f"{split_name}_classification_report.csv")
    np.savetxt(output_dir / f"{split_name}_confusion_matrix.csv", results["confusion_matrix"], fmt="%d", delimiter=",")
    plot_confusion_matrix(results["confusion_matrix"], class_names, output_dir / f"{split_name}_confusion_matrix.png")

    summary = {
        "best_val_acc": float(best_val_acc),
        f"{split_name}_accuracy": float(results["accuracy"]),
        f"{split_name}_macro_f1": float(results["macro_f1"]),
        f"{split_name}_weighted_f1": float(results["weighted_f1"]),
        "num_classes": len(class_names),
        "class_names": class_names,
        "device": str(device),
        "architecture": args.architecture,
        "pretrained": args.pretrained,
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
