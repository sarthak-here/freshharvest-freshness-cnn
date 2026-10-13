from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import train_model
from train_model import build_dataloaders, predict_loader, run_epoch, seed_everything


def main(data_dir: Path, artifacts_dir: Path, epochs: int = 3,
         learning_rate: float = 1e-3, dropout: float = 0.20,
         patience: int = 2):
    seed_everything()
    train_model.IMAGE_SIZE = 224
    classes, datasets, loaders = build_dataloaders(data_dir, batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, len(classes)))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts_dir / "best_resnet50_model.pt"
    history, best_accuracy, stale_epochs = [], -1.0, 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        })
        print(f"Epoch {epoch}: train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy, stale_epochs = val_accuracy, 0
            torch.save({
                "model_state": model.state_dict(), "classes": classes,
                "architecture": "ResNet50", "pretrained": True,
                "frozen_backbone": True, "best_epoch": epoch,
                "best_val_accuracy": best_accuracy, "dropout": dropout,
                "learning_rate": learning_rate,
            }, checkpoint)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    y_true, y_pred = predict_loader(model, loaders["test"], device)
    summary = {
        "architecture": "ResNet50",
        "pretrained": True,
        "frozen_backbone": True,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "weight_decay": 1e-4,
        "early_stopping_patience": patience,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": saved["best_epoch"],
        "best_validation_accuracy": saved["best_val_accuracy"],
        "test_class_accuracy": float((y_true == y_pred).mean()),
        "test_freshness_accuracy": float(((y_true >= 8) == (y_pred >= 8)).mean()),
        "history": history,
    }
    (artifacts_dir / "transfer_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.figure(figsize=(8, 5))
    plot_epochs = [row["epoch"] for row in history]
    plt.plot(plot_epochs, [row["train_accuracy"] for row in history], marker="o", label="Training")
    plt.plot(plot_epochs, [row["val_accuracy"] for row in history], marker="o", label="Validation")
    plt.axhline(0.95, color="black", linestyle=":", label="95% target")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.80, 1.005)
    plt.title("ResNet50 transfer-learning accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "transfer_accuracy.png", dpi=160)
    plt.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/FRUIT-16K"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/transfer"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    main(args.data_dir, args.artifacts_dir, args.epochs,
         args.learning_rate, args.dropout, args.patience)
