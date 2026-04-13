import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = Path("output")


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "mlp_lvl7_label_smoothing",
        "task_name": "MLP Label Smoothing",
        "task_type": "classification",
        "input_type": "tabular",
        "output_type": "multiclass",
        "metrics": ["loss", "accuracy"],
        "description": "Compare multiclass MLP training with and without label smoothing.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_dataset(num_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, 10)).astype(np.float32)

    s0 = 0.8 * X[:, 0] - 1.0 * X[:, 1] + np.sin(X[:, 2])
    s1 = -0.7 * X[:, 3] + 1.1 * X[:, 4] * X[:, 5] + np.cos(X[:, 6])
    s2 = 1.0 * X[:, 7] - 0.8 * (X[:, 8] ** 2) + 0.4 * X[:, 9]

    logits = np.stack([s0, s1, s2], axis=1)
    y = logits.argmax(axis=1).astype(np.int64)

    X += 0.2 * rng.normal(size=X.shape).astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    X, y = _generate_dataset()

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> nn.Module:
    return MLP()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> List[int]:
    model.eval()
    preds_all: List[int] = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            preds_all.extend(preds)

    return preds_all


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_smoothing: float = 0.0,
    epochs: int = 30,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)

        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss_epoch": running_loss / total_seen,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
        })

    return {
        "label_smoothing": label_smoothing,
        "history": history,
        "final_train_metrics": evaluate(model, train_loader, criterion, device),
        "final_val_metrics": evaluate(model, val_loader, criterion, device),
    }


def save_artifacts(results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "label_smoothing_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> int:
    set_seed(42)
    train_loader, val_loader = make_dataloaders()

    set_seed(42)
    vanilla_model = build_model()
    vanilla_results = train(vanilla_model, train_loader, val_loader, label_smoothing=0.0)

    set_seed(42)
    smooth_model = build_model()
    smooth_results = train(smooth_model, train_loader, val_loader, label_smoothing=0.1)

    vanilla_acc = vanilla_results["final_val_metrics"]["accuracy"]
    smooth_acc = smooth_results["final_val_metrics"]["accuracy"]

    vanilla_loss = vanilla_results["final_val_metrics"]["loss"]
    smooth_loss = smooth_results["final_val_metrics"]["loss"]

    results = {
        "vanilla_cross_entropy": vanilla_results,
        "label_smoothing_0_1": smooth_results,
        "accuracy_difference": smooth_acc - vanilla_acc,
        "loss_difference": smooth_loss - vanilla_loss,
    }

    save_artifacts(results)

    checks = {
        "vanilla_accuracy_reasonable": vanilla_acc > 0.80,
        "smoothed_accuracy_reasonable": smooth_acc > 0.78,
        "label_smoothing_not_harmful": smooth_acc > (vanilla_acc - 0.05),
    }

    passed = all(checks.values())

    print(json.dumps({
        "metadata": get_task_metadata(),
        "results": results,
        "checks": checks,
        "passed": passed,
    }, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())