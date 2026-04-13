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
        "task_id": "mlp_lvl5_optimizer_compare",
        "task_name": "MLP Optimizer Comparison",
        "task_type": "classification",
        "input_type": "tabular",
        "output_type": "binary",
        "metrics": ["loss", "accuracy"],
        "description": "Compare SGD, Adam, and RMSprop on the same neural network.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_dataset(num_samples: int = 1800, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_samples, 8)).astype(np.float32)
    score = (
        1.2 * X[:, 0] * X[:, 1]
        + 0.8 * np.sin(X[:, 2])
        - 0.6 * (X[:, 3] ** 2)
        + 0.7 * X[:, 4]
        + 0.2 * rng.normal(size=num_samples)
    )
    y = (score > np.median(score)).astype(np.int64)

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

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_classes: int = 2) -> None:
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
    correct = 0
    total = 0

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
    optimizer_name: str,
    epochs: int = 30,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    device = get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
        "optimizer": optimizer_name,
        "history": history,
        "final_train_metrics": evaluate(model, train_loader, criterion, device),
        "final_val_metrics": evaluate(model, val_loader, criterion, device),
    }


def save_artifacts(results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "optimizer_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> int:
    set_seed(42)
    train_loader, val_loader = make_dataloaders()

    results: Dict[str, Any] = {"runs": {}}

    best_optimizer = None
    best_val_acc = -1.0

    for optimizer_name in ["sgd", "adam", "rmsprop"]:
        set_seed(42)
        model = build_model()
        run_result = train(model, train_loader, val_loader, optimizer_name=optimizer_name)
        results["runs"][optimizer_name] = run_result

        val_acc = run_result["final_val_metrics"]["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_optimizer = optimizer_name

    results["best_optimizer"] = best_optimizer
    results["best_val_accuracy"] = best_val_acc

    save_artifacts(results)

    checks = {
        "all_three_optimizers_ran": len(results["runs"]) == 3,
        "best_val_accuracy_above_threshold": best_val_acc > 0.78,
        "best_optimizer_recorded": best_optimizer is not None,
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