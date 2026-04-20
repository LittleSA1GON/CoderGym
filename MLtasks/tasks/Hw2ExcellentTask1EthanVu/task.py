import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
BATCH_SIZE = 64
EPOCHS = 28
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
SAMPLES_PER_CLASS = 550
OUTPUT_DIR = Path("output")
SUCCESS_ACCURACY_THRESHOLD = 0.68
SUCCESS_MACRO_F1_THRESHOLD = 0.66

BIGQUERY_QUERY = f"""
WITH filtered AS (
    SELECT
        product,
        consumer_complaint_narrative AS content
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE consumer_complaint_narrative IS NOT NULL
      AND LENGTH(consumer_complaint_narrative) BETWEEN 80 AND 2500
      AND product IN (
          'Mortgage',
          'Credit card',
          'Checking or savings account',
          'Student loan'
      )
), ranked AS (
    SELECT
        product,
        content,
        ROW_NUMBER() OVER (
            PARTITION BY product
            ORDER BY FARM_FINGERPRINT(content)
        ) AS row_num
    FROM filtered
)
SELECT product, content
FROM ranked
WHERE row_num <= {SAMPLES_PER_CLASS}
ORDER BY product, row_num
"""


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_LAST_DATA_BUNDLE: Dict[str, Any] = {}


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "hw2excellent_bq_embedding_cfpb_multiclass",
        "task_name": "BigQuery Embeddings CFPB Complaint Product Classifier",
        "task_type": "multiclass_classification",
        "input_type": "text_embeddings",
        "output_type": "product_category",
        "dataset": "bigquery-public-data.cfpb_complaints.complaint_database",
        "bigquery_component": "bigframes.ml.llm.TextEmbeddingGenerator",
        "metrics": ["loss", "accuracy", "macro_f1"],
        "description": (
            "Load complaint narratives from BigQuery BigFrames, generate BigQuery text embeddings, "
            "and train a PyTorch MLP to predict the complaint product class."
        ),
    }


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_bigframes() -> Tuple[Any, Any, Any]:
    try:
        import bigframes  # type: ignore
        import bigframes.pandas as bpd  # type: ignore
        from bigframes.ml.llm import TextEmbeddingGenerator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This task requires the 'bigframes' package and Google Cloud authentication. "
            "Install bigframes, enable the BigQuery, BigQuery Connection, and Vertex AI APIs, "
            "and authenticate with Application Default Credentials before running."
        ) from exc
    return bigframes, bpd, TextEmbeddingGenerator


def _configure_bigframes(bigframes: Any) -> None:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if project_id:
        bigframes.options.bigquery.project = project_id
    bigframes.options.bigquery.location = "US"
    bigframes.options.display.progress_bar = None


def _load_bigquery_dataframe() -> pd.DataFrame:
    bigframes, bpd, TextEmbeddingGenerator = _import_bigframes()
    _configure_bigframes(bigframes)

    raw_df = bpd.read_gbq(BIGQUERY_QUERY)
    raw_df = raw_df[["product", "content"]]

    embedder = TextEmbeddingGenerator(model_name="text-embedding-005")
    embedded = embedder.predict(raw_df[["content"]], max_retries=1)
    successful = (
        (embedded["ml_generate_embedding_status"] == "")
        & (embedded["ml_generate_embedding_result"].str.len() != 0)
    )
    embedded = embedded[successful]

    joined = embedded.join(raw_df[["product"]])
    pdf = joined[["product", "content", "ml_generate_embedding_result"]].to_pandas()
    if pdf.empty:
        raise RuntimeError("No rows were returned from BigQuery embedding generation.")

    pdf = pdf.dropna(subset=["product", "content", "ml_generate_embedding_result"]).reset_index(drop=True)
    return pdf


def _stratified_split_indices(labels: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, List[float], List[List[int]]]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1

    per_class_f1: List[float] = []
    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1.append(float(f1))

    return float(np.mean(per_class_f1)), per_class_f1, confusion.tolist()


def make_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    global _LAST_DATA_BUNDLE

    pdf = _load_bigquery_dataframe()
    embeddings = np.stack(pdf["ml_generate_embedding_result"].map(np.asarray).to_list()).astype(np.float32)

    class_names = sorted(pdf["product"].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    labels = pdf["product"].map(label_to_index).to_numpy(dtype=np.int64)

    train_idx, val_idx = _stratified_split_indices(labels, val_fraction=VAL_FRACTION, seed=SEED)

    x_train = embeddings[train_idx]
    x_val = embeddings[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    bundle = {
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "label_to_index": label_to_index,
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "rows_loaded": int(len(pdf)),
        "standardization": {
            "mean_preview": mean[0, :8].round(6).tolist(),
            "std_preview": std[0, :8].round(6).tolist(),
        },
    }
    _LAST_DATA_BUNDLE = bundle
    return train_loader, val_loader, bundle


def build_model(input_dim: int, num_classes: int) -> nn.Module:
    return EmbeddingMLP(input_dim=input_dim, num_classes=num_classes)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            y_true_parts.append(yb.cpu().numpy())
            y_pred_parts.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    accuracy = float((y_true == y_pred).mean())
    macro_f1, per_class_f1, confusion = _macro_f1_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=int(_LAST_DATA_BUNDLE["num_classes"]),
    )

    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion,
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> List[int]:
    model.eval()
    outputs: List[int] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            outputs.extend(preds)
    return outputs


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
    device = get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_val_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    history: List[Dict[str, Any]] = []
    patience = 6
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "train_loss_epoch": float(running_loss / max(seen, 1)),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    final_train_metrics = evaluate(model, train_loader, device)
    final_val_metrics = evaluate(model, val_loader, device)

    return {
        "history": history,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "epochs_completed": len(history),
    }


def save_artifacts(results: Dict[str, Any], model: nn.Module) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def main() -> int:
    set_seed(SEED)
    try:
        train_loader, val_loader, bundle = make_dataloaders()
        model = build_model(input_dim=bundle["input_dim"], num_classes=bundle["num_classes"])
        training_results = train(model, train_loader, val_loader)

        results = {
            "metadata": get_task_metadata(),
            "config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "samples_per_class": SAMPLES_PER_CLASS,
            },
            "data_summary": bundle,
            "training": training_results,
        }

        val_metrics = training_results["final_val_metrics"]
        checks = {
            "loaded_four_classes": bundle["num_classes"] == 4,
            "enough_rows_loaded": bundle["rows_loaded"] >= 4 * int(SAMPLES_PER_CLASS * 0.85),
            "val_accuracy_ok": val_metrics["accuracy"] >= SUCCESS_ACCURACY_THRESHOLD,
            "val_macro_f1_ok": val_metrics["macro_f1"] >= SUCCESS_MACRO_F1_THRESHOLD,
        }
        passed = all(checks.values())
        results["checks"] = checks
        results["passed"] = passed

        save_artifacts(results, model)
        print(json.dumps(results, indent=2))
        return 0 if passed else 1
    except Exception as exc:
        error_payload = {
            "metadata": get_task_metadata(),
            "passed": False,
            "error": str(exc),
        }
        print(json.dumps(error_payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())