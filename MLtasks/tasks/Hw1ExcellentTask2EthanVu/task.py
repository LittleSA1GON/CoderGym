"""
PyTorch Logistic Regression task using Google BigQuery.

What this script does:
1. Connects to BigQuery
2. Loads Chicago Taxi Trips data
3. Creates a binary label: tipped or not tipped
4. Trains a PyTorch logistic regression model
5. Evaluates the model with accuracy and F1
6. Exits with code 0 if successful, otherwise exits with code 1
"""

# -----------------------------
# Section 1: Imports
# -----------------------------
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from google.cloud import bigquery


# -----------------------------
# Section 2: Configuration
# -----------------------------
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 200
SUCCESS_ACCURACY_THRESHOLD = 0.60
OUTPUT_DIR = "logistic_regression_outputs"


# -----------------------------
# Section 3: Reproducibility
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Section 4: Metadata
# -----------------------------
def get_task_metadata():
    return {
        "task_id": "logreg_bigquery_chicago_taxi_simple",
        "dataset": "bigquery-public-data.chicago_taxi_trips.taxi_trips",
        "target": "tipped_anything",
        "model": "PyTorch Logistic Regression",
        "description": "Predict whether a taxi trip received a tip using BigQuery data."
    }


# -----------------------------
# Section 5: Load data from BigQuery
# -----------------------------
# We create a binary label:
# tipped_anything = 1 if tips > 0 else 0
def load_data_from_bigquery():
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

    query = """
    SELECT
      IF(tips > 0, 1, 0) AS tipped_anything,
      CAST(trip_miles AS FLOAT64) AS trip_miles,
      CAST(fare AS FLOAT64) AS fare
    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE tips IS NOT NULL
      AND trip_miles IS NOT NULL
      AND fare IS NOT NULL
      AND trip_miles > 0
      AND fare > 0
    LIMIT 10000
    """

    df = client.query(query).result().to_dataframe()
    return df


# -----------------------------
# Section 6: Train/validation split
# -----------------------------
def split_data(df):
    train_df = df.sample(frac=0.8, random_state=SEED)
    val_df = df.drop(train_df.index)
    return train_df, val_df


# -----------------------------
# Section 7: Feature preprocessing
# -----------------------------
# We standardize numeric features and convert them to tensors.
def prepare_features(train_df, val_df):
    feature_cols = ["trip_miles", "fare"]
    target_col = "tipped_anything"

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)

    X_train = (train_df[feature_cols] - mean) / std
    X_val = (val_df[feature_cols] - mean) / std

    y_train = train_df[[target_col]]
    y_val = val_df[[target_col]]

    X_train = torch.tensor(X_train.to_numpy(dtype=np.float32))
    X_val = torch.tensor(X_val.to_numpy(dtype=np.float32))
    y_train = torch.tensor(y_train.to_numpy(dtype=np.float32))
    y_val = torch.tensor(y_val.to_numpy(dtype=np.float32))

    preprocess_info = {
        "feature_cols": feature_cols,
        "mean": mean.to_dict(),
        "std": std.to_dict()
    }

    return X_train, X_val, y_train, y_val, preprocess_info


# -----------------------------
# Section 8: Build the model
# -----------------------------
# Logistic regression in PyTorch can still use nn.Linear.
# The difference is in the loss function and output interpretation.
def build_model(input_dim):
    return nn.Linear(input_dim, 1)


# -----------------------------
# Section 9: Train the model
# -----------------------------
# We use BCEWithLogitsLoss because:
# - model outputs a raw score (logit)
# - this loss is designed for binary classification
def train_model(model, X_train, y_train):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = []

    for epoch in range(EPOCHS):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {loss.item():.4f}")

    return history


# -----------------------------
# Section 10: Evaluation metrics
# -----------------------------
# For classification:
# - sigmoid converts logits to probabilities
# - threshold 0.5 converts probabilities to 0/1 predictions
# - we compute accuracy, precision, recall, and F1
def evaluate_model(model, X_val, y_val):
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    y_true = y_val.numpy().reshape(-1)
    y_pred = preds.numpy().reshape(-1)

    accuracy = float(np.mean(y_true == y_pred))

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -----------------------------
# Section 11: Save outputs
# -----------------------------
def save_outputs(model, history, metrics, metadata, preprocess_info):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "preprocess.json"), "w") as f:
        json.dump(preprocess_info, f, indent=2)


# -----------------------------
# Section 12: Main workflow
# -----------------------------
def main():
    try:
        set_seed(SEED)

        metadata = get_task_metadata()
        print("Task metadata:")
        print(json.dumps(metadata, indent=2))

        # Load BigQuery data
        df = load_data_from_bigquery()
        print(f"Loaded {len(df)} rows from BigQuery")

        # Split into train and validation
        train_df, val_df = split_data(df)

        # Prepare features
        X_train, X_val, y_train, y_val, preprocess_info = prepare_features(train_df, val_df)

        # Build and train model
        model = build_model(input_dim=X_train.shape[1])
        history = train_model(model, X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        print("Validation metrics:")
        print(json.dumps(metrics, indent=2))

        # Save outputs
        save_outputs(model, history, metrics, metadata, preprocess_info)

        # Self-verification rule
        if metrics["accuracy"] > SUCCESS_ACCURACY_THRESHOLD:
            print("Task succeeded.")
            sys.exit(0)
        else:
            print("Task failed: accuracy too low.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# -----------------------------
# Section 13: Script entry point
# -----------------------------
if __name__ == "__main__":
    main()