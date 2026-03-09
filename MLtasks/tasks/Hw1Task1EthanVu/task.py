import os
import sys
import csv
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Auto-match folder name (so you can rename the folder without editing code)
TASK_ID = os.path.basename(os.path.dirname(__file__))

# UCI raw file (CSV-like). Header shows columns including Compressive Strength. :contentReference[oaicite:1]{index=1}
UCI_SLUMP_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"


def get_task_metadata():
    return {
        "task_type": "regression",
        "model_type": "linear_regression",
        "input_dim": 7,   # Cement, Slag, Fly ash, Water, SP, Coarse, Fine
        "output_dim": 1,
        "description": "Linear regression on UCI Concrete Slump Test using Adam + early stopping (predict 28-day compressive strength).",
    }


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_output_dir():
    preferred = f"/Developer/AIserver/output/tasks/{TASK_ID}"
    fallback = os.path.join("output", "tasks", TASK_ID)
    env_base = os.environ.get("CODERGYM_OUTPUT_DIR", "").strip()
    if env_base:
        preferred = os.path.join(env_base, "tasks", TASK_ID)

    for path in (preferred, fallback):
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".write_test")
            with open(testfile, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(testfile)
            return path
        except Exception:
            pass
    return fallback


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return (1.0 - (ss_res / (ss_tot + 1e-12))).item()


def _download_if_missing(url: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return
    print(f"Downloading dataset:\n  {url}\n-> {local_path}")
    urllib.request.urlretrieve(url, local_path)


def _load_slump_dataset(local_path: str):
    """
    File has header:
      No,Cement,Slag,Fly ash,Water,SP,Coarse Aggr.,Fine Aggr.,SLUMP(cm),FLOW(cm),Compressive Strength (28-day)(Mpa)
    We'll use columns 1..7 as X (7 features), and column 10 as y.
    :contentReference[oaicite:2]{index=2}
    """
    X_list, y_list = [], []
    with open(local_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 11:
                continue
            # row[0] is "No" (index)
            x = [float(v) for v in row[1:8]]    # 7 inputs
            y = float(row[10])                  # compressive strength
            X_list.append(x)
            y_list.append(y)

    X = np.array(X_list, dtype=np.float32)              # [N,7]
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y


def make_dataloaders(batch_size: int = 32, train_ratio: float = 0.8):
    """
    Newbie-friendly:
    - download once
    - train/val split
    - standardize using TRAIN stats only
    """
    data_path = os.path.join("MLtasks", "data", "uci_slump", "slump_test.data")
    _download_if_missing(UCI_SLUMP_DATA_URL, data_path)

    X, y = _load_slump_dataset(data_path)
    n = X.shape[0]

    idx = np.random.permutation(n)
    n_train = int(train_ratio * n)
    tr, va = idx[:n_train], idx[n_train:]

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sigma
    X_va = (X_va - mu) / sigma

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, {"data_path": data_path, "train_mu": mu, "train_sigma": sigma}


def build_model(device=None):
    model = torch.nn.Linear(get_task_metadata()["input_dim"], 1)
    if device is not None:
        model.to(device)
    return model


def train(model, train_loader, val_loader, device,
          lr: float = 0.05, weight_decay: float = 1e-3,
          max_epochs: int = 2000, patience: int = 80):
    """
    Feature additions:
    - Adam optimizer
    - Early stopping on validation MSE
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_m = evaluate(model, val_loader, device)
        if val_m["mse"] < best_val - 1e-6:
            best_val = val_m["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 250 == 0 or epoch == 1:
            tr_m = evaluate(model, train_loader, device)
            print(f"Epoch {epoch:4d} | train_mse={tr_m['mse']:.3f} val_mse={val_m['mse']:.3f}")

        if bad >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model, data_loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds.append(model(xb))
            ys.append(yb)

    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(preds, dim=0)

    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    r2 = _r2(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2}


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        return model(X.to(device)).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"metadata": get_task_metadata(), "state_dict": model.state_dict(), "metrics": metrics},
        os.path.join(output_dir, "model_and_metrics.pt"),
    )


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    out_dir = _resolve_output_dir()

    print("=" * 70)
    print("TASK:", TASK_ID)
    print(get_task_metadata()["description"])
    print("Device:", device)
    print("Output:", out_dir)
    print("=" * 70)

    train_loader, val_loader, extras = make_dataloaders()
    print("Data file:", extras["data_path"])

    model = build_model(device=device)
    model = train(model, train_loader, val_loader, device=device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print("TRAIN:", train_m)
    print("VAL  :", val_m)

    # Small dataset -> keep thresholds modest but meaningful
    exit_code = 0
    try:
        assert val_m["r2"] > 0.35, f"Val R2 too low: {val_m['r2']:.4f}"
        assert val_m["mse"] < 140.0, f"Val MSE too high: {val_m['mse']:.3f}"
    except AssertionError as e:
        print("FAIL:", str(e))
        exit_code = 1

    save_artifacts(model, {"train": train_m, "val": val_m}, out_dir)
    sys.exit(exit_code)