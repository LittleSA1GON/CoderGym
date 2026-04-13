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
        "task_id": "mlp_lvl8_early_stopping",
        "task_name": "MLP Early Stopping",
        "task_type": "classification",
        "input_type": "tabular",
        "output_type": "binary",
        "metrics": ["loss", "accuracy"],
        "description": "Train a deeper MLP with early stopping and recover the best checkpoint.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_dataset(num_samples: int = 2200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, 20)).astype(np.float32)

    score = (
        1.1 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.7 * X[:, 2] * X[:, 3]
        - 0.5 * (X[:, 4] ** 2)
        + 0.9 * rng.normal(size=num_samples)
    )
    y = (score > np.median(score)).astype(np.int64)

    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    X, y = _generate_dataset()

    indices = np.random.permutation(len(X))
    split = int(0.75 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> nn.Module:
    return DeepMLP()


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


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 6,
) -> Dict[str, Any]:
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    history = []

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

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

        if val_metrics["accuracy"] > best_val_acc + 1e-6:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    final_val_metrics = evaluate(model, val_loader, criterion, device)

    best_model = build_model().to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    recovered_best_val_metrics = evaluate(best_model, val_loader, criterion, device)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "final_val_metrics": final_val_metrics,
        "recovered_best_val_metrics": recovered_best_val_metrics,
        "recorded_best_val_accuracy": best_val_acc,
    }


def save_artifacts(results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "early_stopping_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> int:
    set_seed(42)
    train_loader, val_loader = make_dataloaders()

    model = build_model()
    results = train(model, train_loader, val_loader)

    save_artifacts(results)

    recovered_best_acc = results["recovered_best_val_metrics"]["accuracy"]
    final_acc = results["final_val_metrics"]["accuracy"]

    checks = {
        "early_stopping_triggered_before_max_epochs": results["stopped_epoch"] < 60,
        "best_checkpoint_recovered": abs(recovered_best_acc - results["recorded_best_val_accuracy"]) < 1e-6,
        "best_checkpoint_not_worse_than_final": recovered_best_acc >= final_acc,
        "validation_accuracy_reasonable": recovered_best_acc > 0.72,
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