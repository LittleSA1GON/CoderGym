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
        "task_id": "mlp_lvl6_feature_engineering",
        "task_name": "MLP Feature Engineering",
        "task_type": "classification",
        "input_type": "tabular",
        "output_type": "binary",
        "metrics": ["loss", "accuracy"],
        "description": "Compare raw features against engineered nonlinear features.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_raw_dataset(num_samples: int = 1600, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.5, 2.5, size=(num_samples, 4)).astype(np.float32)
    score = (
        X[:, 0] ** 2
        + 0.8 * np.sin(2.0 * X[:, 1])
        - 0.7 * X[:, 2] * X[:, 3]
        + 0.2 * rng.normal(size=num_samples)
    )
    y = (score > 0.8).astype(np.int64)

    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y


def _engineer_features(X: np.ndarray) -> np.ndarray:
    engineered = np.concatenate(
        [
            X,
            X ** 2,
            np.sin(X),
            np.cos(X),
            (X[:, [0]] * X[:, [1]]),
            (X[:, [2]] * X[:, [3]]),
        ],
        axis=1,
    ).astype(np.float32)

    engineered = (engineered - engineered.mean(axis=0, keepdims=True)) / (
        engineered.std(axis=0, keepdims=True) + 1e-6
    )
    return engineered


def make_dataloaders(batch_size: int = 64, use_engineered_features: bool = False) -> Tuple[DataLoader, DataLoader, int]:
    X, y = _generate_raw_dataset()

    if use_engineered_features:
        X = _engineer_features(X)

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X.shape[1]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(input_dim: int) -> nn.Module:
    return MLP(input_dim=input_dim)


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
    outputs: List[int] = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            outputs.extend(preds)

    return outputs


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 35,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
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
        "history": history,
        "final_train_metrics": evaluate(model, train_loader, criterion, device),
        "final_val_metrics": evaluate(model, val_loader, criterion, device),
    }


def save_artifacts(results: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "feature_engineering_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> int:
    set_seed(42)

    raw_train_loader, raw_val_loader, raw_input_dim = make_dataloaders(use_engineered_features=False)
    eng_train_loader, eng_val_loader, eng_input_dim = make_dataloaders(use_engineered_features=True)

    set_seed(42)
    raw_model = build_model(raw_input_dim)
    raw_results = train(raw_model, raw_train_loader, raw_val_loader)

    set_seed(42)
    eng_model = build_model(eng_input_dim)
    eng_results = train(eng_model, eng_train_loader, eng_val_loader)

    raw_val_acc = raw_results["final_val_metrics"]["accuracy"]
    eng_val_acc = eng_results["final_val_metrics"]["accuracy"]

    results = {
        "raw_features": raw_results,
        "engineered_features": eng_results,
        "raw_input_dim": raw_input_dim,
        "engineered_input_dim": eng_input_dim,
        "validation_accuracy_gain": eng_val_acc - raw_val_acc,
    }

    save_artifacts(results)

    checks = {
        "raw_model_reasonable": raw_val_acc > 0.72,
        "engineered_model_reasonable": eng_val_acc > 0.80,
        "engineered_beats_raw": eng_val_acc > raw_val_acc,
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