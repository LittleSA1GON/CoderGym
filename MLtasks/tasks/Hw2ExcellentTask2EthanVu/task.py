import copy
import json
import os
import random
import re
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
EPOCHS = 36
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
SAMPLES_PER_LABEL = 1200
OUTPUT_DIR = Path("output")
SUCCESS_ACCURACY_THRESHOLD = 0.62
SUCCESS_F1_THRESHOLD = 0.62

BIGQUERY_QUERY = f"""
WITH base AS (
    SELECT
        title,
        REGEXP_REPLACE(body, r'<[^>]+>', ' ') AS body_text,
        tags,
        CAST(accepted_answer_id IS NOT NULL AS INT64) AS label,
        ARRAY_LENGTH(REGEXP_EXTRACT_ALL(tags, r'<[^>]+>')) AS tag_count
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    WHERE creation_date >= '2018-01-01'
      AND title IS NOT NULL
      AND body IS NOT NULL
      AND tags IS NOT NULL
      AND tags LIKE '%python%'
      AND LENGTH(REGEXP_REPLACE(body, r'<[^>]+>', ' ')) BETWEEN 120 AND 2000
), positives AS (
    SELECT *
    FROM base
    WHERE label = 1
    ORDER BY FARM_FINGERPRINT(CONCAT(title, body_text))
    LIMIT {SAMPLES_PER_LABEL}
), negatives AS (
    SELECT *
    FROM base
    WHERE label = 0
    ORDER BY FARM_FINGERPRINT(CONCAT(title, body_text))
    LIMIT {SAMPLES_PER_LABEL}
)
SELECT * FROM positives
UNION ALL
SELECT * FROM negatives
"""


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_LAST_DATA_BUNDLE: Dict[str, Any] = {}


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "hw2excellent_bq_gemini_stackoverflow_binary",
        "task_name": "BigQuery Gemini Stack Overflow Accepted Answer Predictor",
        "task_type": "binary_classification",
        "input_type": "structured_llm_features",
        "output_type": "accepted_answer_label",
        "dataset": "bigquery-public-data.stackoverflow.posts_questions",
        "bigquery_component": "bigframes.ml.llm.GeminiTextGenerator",
        "metrics": ["loss", "accuracy", "precision", "recall", "f1"],
        "description": (
            "Load Stack Overflow questions from BigQuery BigFrames, use Gemini to create structured "
            "question-quality features, and train a PyTorch classifier to predict whether the question "
            "has an accepted answer."
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
        from bigframes.ml.llm import GeminiTextGenerator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This task requires the 'bigframes' package and Google Cloud authentication. "
            "Install bigframes, enable the BigQuery, BigQuery Connection, and Vertex AI APIs, "
            "and authenticate with Application Default Credentials before running."
        ) from exc
    return bigframes, bpd, GeminiTextGenerator


def _configure_bigframes(bigframes: Any) -> None:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if project_id:
        bigframes.options.bigquery.project = project_id
    bigframes.options.bigquery.location = "US"
    bigframes.options.display.progress_bar = None


def _load_bigquery_dataframe() -> pd.DataFrame:
    bigframes, bpd, GeminiTextGenerator = _import_bigframes()
    _configure_bigframes(bigframes)

    source_df = bpd.read_gbq(BIGQUERY_QUERY)
    source_df = source_df[["title", "body_text", "tags", "label", "tag_count"]]

    gemini = GeminiTextGenerator(model_name="gemini-2.0-flash-001")
    llm_df = gemini.predict(
        source_df[["title", "body_text", "tags"]],
        prompt=[
            "Read this Stack Overflow question and estimate answerability features.",
            "Title:", source_df["title"],
            "Body:", source_df["body_text"],
            "Tags:", source_df["tags"],
            (
                "Return structured values only. Score clarity and specificity from 0.0 to 1.0. "
                "Use booleans for the other fields."
            ),
        ],
        output_schema={
            "clarity_score": "float64",
            "specificity_score": "float64",
            "contains_error_message": "bool",
            "contains_minimal_example": "bool",
            "debugging_request": "bool",
            "conceptual_question": "bool",
        },
    )

    combined = source_df.join(
        llm_df[
            [
                "clarity_score",
                "specificity_score",
                "contains_error_message",
                "contains_minimal_example",
                "debugging_request",
                "conceptual_question",
            ]
        ]
    )
    pdf = combined.to_pandas().reset_index(drop=True)
    if pdf.empty:
        raise RuntimeError("No rows were returned from BigQuery Gemini feature generation.")
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


def _count_code_markers(text: str) -> int:
    text = text or ""
    return text.count("```") + text.count("`")


def _count_question_marks(text: str) -> int:
    return int((text or "").count("?"))


def _contains_trace_words(text: str) -> int:
    lowered = (text or "").lower()
    patterns = ["traceback", "exception", "error", "stack trace", "warning"]
    return int(any(pattern in lowered for pattern in patterns))


def _build_feature_matrix(pdf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pdf = pdf.copy()
    pdf["body_text"] = pdf["body_text"].fillna("")
    pdf["title"] = pdf["title"].fillna("")
    pdf["tags"] = pdf["tags"].fillna("")

    bool_cols = [
        "contains_error_message",
        "contains_minimal_example",
        "debugging_request",
        "conceptual_question",
    ]
    for col in bool_cols:
        pdf[col] = pdf[col].fillna(False).astype(bool).astype(np.float32)

    score_cols = ["clarity_score", "specificity_score"]
    for col in score_cols:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    pdf["title_len"] = pdf["title"].str.len().astype(np.float32)
    pdf["body_len"] = pdf["body_text"].str.len().astype(np.float32)
    pdf["tag_count"] = pd.to_numeric(pdf["tag_count"], errors="coerce").fillna(0).astype(np.float32)
    pdf["question_mark_count"] = pdf["body_text"].map(_count_question_marks).astype(np.float32)
    pdf["code_marker_count"] = pdf["body_text"].map(_count_code_markers).astype(np.float32)
    pdf["trace_word_flag"] = pdf["body_text"].map(_contains_trace_words).astype(np.float32)
    pdf["title_has_how_to"] = pdf["title"].str.lower().str.contains(r"\bhow\b", regex=True).fillna(False).astype(np.float32)
    pdf["body_has_version"] = pdf["body_text"].str.lower().str.contains(r"python\s*\d|\d+\.\d+", regex=True).fillna(False).astype(np.float32)

    feature_cols = [
        "clarity_score",
        "specificity_score",
        "contains_error_message",
        "contains_minimal_example",
        "debugging_request",
        "conceptual_question",
        "title_len",
        "body_len",
        "tag_count",
        "question_mark_count",
        "code_marker_count",
        "trace_word_flag",
        "title_has_how_to",
        "body_has_version",
    ]

    X = pdf[feature_cols].to_numpy(dtype=np.float32)
    y = pd.to_numeric(pdf["label"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    return X, y, feature_cols


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def make_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    global _LAST_DATA_BUNDLE

    pdf = _load_bigquery_dataframe()
    X, y, feature_cols = _build_feature_matrix(pdf)

    train_idx, val_idx = _stratified_split_indices(y.astype(np.int64), val_fraction=VAL_FRACTION, seed=SEED)
    x_train = X[train_idx]
    x_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    bundle = {
        "input_dim": int(x_train.shape[1]),
        "feature_names": feature_cols,
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "rows_loaded": int(len(pdf)),
        "label_balance": {
            "positive_fraction": float(y.mean()),
            "negative_fraction": float(1.0 - y.mean()),
        },
        "standardization": {
            "mean_preview": mean[0, :6].round(6).tolist(),
            "std_preview": std[0, :6].round(6).tolist(),
        },
    }
    _LAST_DATA_BUNDLE = bundle
    return train_loader, val_loader, bundle


def build_model(input_dim: int) -> nn.Module:
    return FeatureMLP(input_dim=input_dim)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
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
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            y_true_parts.append(yb.cpu().numpy().reshape(-1))
            y_pred_parts.append(preds.cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true_parts).astype(np.int64)
    y_pred = np.concatenate(y_pred_parts).astype(np.int64)
    metric_payload = _binary_metrics(y_true, y_pred)
    metric_payload["loss"] = float(total_loss / max(total, 1))
    return metric_payload


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> List[int]:
    model.eval()
    outputs: List[int] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().view(-1).tolist()
            outputs.extend(preds)
    return outputs


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
    device = get_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
    history: List[Dict[str, Any]] = []
    patience = 8
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
        scheduler.step(val_metrics["f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss_epoch": float(running_loss / max(seen, 1)),
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
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
        model = build_model(input_dim=bundle["input_dim"])
        training_results = train(model, train_loader, val_loader)

        results = {
            "metadata": get_task_metadata(),
            "config": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "samples_per_label": SAMPLES_PER_LABEL,
            },
            "data_summary": bundle,
            "training": training_results,
        }

        val_metrics = training_results["final_val_metrics"]
        checks = {
            "balanced_label_sample": abs(bundle["label_balance"]["positive_fraction"] - 0.5) <= 0.08,
            "enough_rows_loaded": bundle["rows_loaded"] >= int(SAMPLES_PER_LABEL * 1.8),
            "val_accuracy_ok": val_metrics["accuracy"] >= SUCCESS_ACCURACY_THRESHOLD,
            "val_f1_ok": val_metrics["f1"] >= SUCCESS_F1_THRESHOLD,
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