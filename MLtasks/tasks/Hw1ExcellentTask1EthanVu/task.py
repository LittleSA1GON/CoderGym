"""
Beginner-friendly PyTorch Linear Regression task using Google BigQuery.

What this script does:
1. Connects to BigQuery
2. Loads Austin Bikeshare trip data
3. Uses a few simple numeric features
4. Trains a PyTorch linear regression model
5. Evaluates the model with MSE and R^2
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
# These are basic settings you can change later if needed.
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 200
SUCCESS_MSE_THRESHOLD = 200.0   # simple success rule for homework
OUTPUT_DIR = "linear_regression_outputs"


# -----------------------------
# Section 3: Reproducibility
# -----------------------------
# Setting seeds helps make results more repeatable.
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Section 4: Metadata
# -----------------------------
# This describes the task in a structured way.
def get_task_metadata():
    return {
        "task_id": "linreg_bigquery_austin_bikeshare_simple",
        "dataset": "bigquery-public-data.austin_bikeshare.bikeshare_trips",
        "target": "duration_minutes",
        "model": "PyTorch Linear Regression",
        "description": "Predict trip duration using simple numeric features from BigQuery."
    }


# -----------------------------
# Section 5: Load data from BigQuery
# -----------------------------
# This is the required BigQuery loading step.
# We only use a few numeric columns to keep it beginner-friendly.
def load_data_from_bigquery():
    client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

    query = """
    SELECT
      CAST(duration_minutes AS FLOAT64) AS duration_minutes,
      EXTRACT(HOUR FROM start_time) AS start_hour,
      EXTRACT(DAYOFWEEK FROM start_time) AS day_of_week
    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
    WHERE duration_minutes IS NOT NULL
      AND start_time IS NOT NULL
      AND duration_minutes BETWEEN 1 AND 30
    LIMIT 10000
    """

    df = client.query(query).result().to_dataframe()
    return df


# -----------------------------
# Section 6: Train/validation split
# -----------------------------
# We split the dataset so we can train on one part
# and test performance on a different part.
def split_data(df):
    train_df = df.sample(frac=0.8, random_state=SEED)
    val_df = df.drop(train_df.index)
    return train_df, val_df


# -----------------------------
# Section 7: Feature preprocessing
# -----------------------------
# PyTorch needs numeric tensors.
# We:
# 1. choose input columns
# 2. standardize them
# 3. convert them to tensors
def prepare_features(train_df, val_df):
    feature_cols = ["start_hour", "day_of_week"]
    target_col = "duration_minutes"

    # Compute mean and std from training data only
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)

    # Standardize train and validation features
    X_train = (train_df[feature_cols] - mean) / std
    X_val = (val_df[feature_cols] - mean) / std

    # Extract targets
    y_train = train_df[[target_col]]
    y_val = val_df[[target_col]]

    # Convert to PyTorch tensors
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
# A single linear layer is the PyTorch version of linear regression.
def build_model(input_dim):
    return nn.Linear(input_dim, 1)


# -----------------------------
# Section 9: Train the model
# -----------------------------
# We use:
# - MSELoss for regression
# - SGD optimizer for simplicity
def train_model(model, X_train, y_train):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = []

    for epoch in range(EPOCHS):
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
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
# MSE = average squared error
# R^2 = how much variance the model explains
def evaluate_model(model, X_val, y_val):
    with torch.no_grad():
        preds = model(X_val)

    preds_np = preds.numpy().reshape(-1)
    y_np = y_val.numpy().reshape(-1)

    mse = float(np.mean((preds_np - y_np) ** 2))

    ss_res = float(np.sum((y_np - preds_np) ** 2))
    ss_tot = float(np.sum((y_np - np.mean(y_np)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "r2": r2
    }


# -----------------------------
# Section 11: Save outputs
# -----------------------------
# Save model weights and metrics for review.
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
# This runs the complete ML pipeline from start to finish.
def main():
    try:
        set_seed(SEED)

        metadata = get_task_metadata()
        print("Task metadata:")
        print(json.dumps(metadata, indent=2))
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
        # If MSE is reasonably low, consider task successful
        if metrics["mse"] < SUCCESS_MSE_THRESHOLD:
            print("Task succeeded.")
            sys.exit(0)
        else:
            print("Task failed: MSE too high.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()