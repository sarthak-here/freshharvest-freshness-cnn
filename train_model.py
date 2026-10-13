from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


SEED = 42
IMAGE_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IndexedImageDataset(Dataset):
    def __init__(self, base: ImageFolder, indices: list[int], transform) -> None:
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        path, label = self.base.samples[self.indices[position]]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def stratified_indices(targets: list[int], train_ratio=0.70, val_ratio=0.15):
    rng = np.random.default_rng(SEED)
    targets_array = np.asarray(targets)
    train, val, test = [], [], []
    for class_id in sorted(np.unique(targets_array)):
        indices = np.flatnonzero(targets_array == class_id)
        rng.shuffle(indices)
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        train.extend(indices[:train_end])
        val.extend(indices[train_end:val_end])
        test.extend(indices[val_end:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_dataloaders(data_dir: Path, batch_size: int = 64):
    base = ImageFolder(data_dir)
    train_idx, val_idx, test_idx = stratified_indices(base.targets)

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    datasets = {
        "train": IndexedImageDataset(base, train_idx, eval_transform),
        "val": IndexedImageDataset(base, val_idx, eval_transform),
        "test": IndexedImageDataset(base, test_idx, eval_transform),
    }
    pin_memory = torch.cuda.is_available()
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=pin_memory,
        )
        for name, dataset in datasets.items()
    }
    return base.classes, datasets, loaders


class FreshHarvestCNN(nn.Module):
    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_items += labels.size(0)
    return total_loss / total_items, total_correct / total_items


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    true_labels, predicted_labels = [], []
    for images, labels in loader:
        predictions = model(images.to(device, non_blocking=True)).argmax(1).cpu()
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())
    return np.asarray(true_labels), np.asarray(predicted_labels)


def predict_image(image_path: Path, checkpoint_path: Path = Path("artifacts/best_model.pt")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FreshHarvestCNN(len(saved["classes"])).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((saved["image_size"], saved["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(saved["mean"], saved["std"]),
    ])
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0]
    index = int(probabilities.argmax())
    label = saved["classes"][index]
    condition, produce = label.split("_", 1)
    return {
        "produce": produce,
        "freshness": "Fresh" if condition == "F" else "Spoiled",
        "class_label": label,
        "confidence": float(probabilities[index]),
    }


def save_training_plot(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [r["train_accuracy"] for r in history], label="Train")
    axes[1].plot(epochs, [r["val_accuracy"] for r in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.01))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train(
    data_dir: Path,
    artifacts_dir: Path,
    epochs: int = 8,
    batch_size: int = 64,
    epoch_candidates: tuple[int, ...] = (3, 5, 8),
):
    seed_everything()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    classes, datasets, loaders = build_dataloaders(data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreshHarvestCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    best_val_accuracy = -1.0
    checkpoint = artifacts_dir / "best_model.pt"
    print(f"Device: {device} | Classes: {len(classes)} | Split sizes: "
          f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}")

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "mean": MEAN,
                "std": STD,
                "best_val_accuracy": best_val_accuracy,
            }, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    y_true, y_pred = predict_loader(model, loaders["test"], device)
    test_accuracy = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    freshness_true = y_true >= 8
    freshness_pred = y_pred >= 8
    freshness_accuracy = float((freshness_true == freshness_pred).mean())

    metrics = {
        "device": str(device),
        "class_names": classes,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "best_validation_accuracy": best_val_accuracy,
        "test_class_accuracy": test_accuracy,
        "test_freshness_accuracy": freshness_accuracy,
        "epoch_experiments": [
            row for row in history if row["epoch"] in epoch_candidates
        ],
        "selected_epoch": int(max(history, key=lambda row: row["val_accuracy"])["epoch"]),
        "regularization_used": False,
        "transfer_learning_used": False,
        "classification_report": report,
        "history": history,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_plot(history, artifacts_dir / "training_history.png")

    plt.figure(figsize=(13, 11))
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.title("FreshHarvest test confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test 16-class accuracy: {test_accuracy:.4f}")
    print(f"Test fresh/spoiled accuracy: {freshness_accuracy:.4f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FreshHarvest freshness-inspection CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/FRUIT-16K"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.artifacts_dir, args.epochs, args.batch_size)

