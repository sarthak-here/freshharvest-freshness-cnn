from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_CHECKPOINT = Path("artifacts/transfer/best_resnet50_model.pt")


def load_transfer_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    """Load the saved ResNet50 checkpoint on CPU for portable inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    dropout = float(checkpoint.get("dropout", 0.20))

    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def prepare_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def predict(image: Image.Image, model, classes: list[str], top_k: int = 3):
    probabilities = model(prepare_image(image)).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(classes)))

    predictions = []
    for probability, index in zip(values.tolist(), indices.tolist()):
        class_label = classes[index]
        condition, produce = class_label.split("_", 1)
        predictions.append({
            "class_label": class_label,
            "produce": produce,
            "freshness": "Fresh" if condition == "F" else "Spoiled",
            "confidence": probability,
        })
    return predictions

