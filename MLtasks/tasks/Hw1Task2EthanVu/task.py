import os
import sys
import csv
import zipfile
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

TASK_ID = os.path.basename(os.path.dirname(__file__))

# UCI lists winequality-red.csv in the dataset files. :contentReference[oaicite:6]{index=6}
UCI_WINE_ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip"


def get_task_metadata():
    return {
        "task_type": "regression",
        "model_type": "linear_regression",
        "input_dim": 11,
        "output_dim": 1,
        "description": "Linear regression on UCI Wine Quality (red) using SGD+momentum + StepLR scheduler.",
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


def _download_and_extract_wine_red(data_dir: str):
    """
    Newbie-friendly: download once, cache locally.
    We download the UCI zip (Wine+Quality) and extract winequality-red.csv.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "wine_quality.zip")
    red_csv_path = os.path.join(data_dir, "winequality-red.csv")

    if os.path.exists(red_csv_path):
        return red_csv_path

    if not os.path.exists(zip_path):
        print("Downloading UCI Wine Quality zip...")
        urllib.request.urlretrieve(UCI_WINE_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains winequality-red.csv (as listed by UCI). :contentReference[oaicite:7]{index=7}
        zf.extract("winequality-red.csv", path=data_dir)

    return red_csv_path


def _load_wine_red_csv(csv_path: str):
    """
    UCI winequality-red.csv uses ';' delimiter and has a header row.
    Features: 11 columns, target: 'quality' (0..10).
    """
    X = []
    y = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        # last column is quality
        for row in reader:
            vals = [float(v) for v in row]
            X.append(vals[:-1])
            y.append(vals[-1])

    X = np.array(X, dtype=np.float32)          # [N, 11]
    y = np.array(y, dtype=np.float32).reshape(-1, 1)  # [N, 1]
    return X, y


def make_dataloaders(batch_size: int = 128, train_ratio: float = 0.8):
    data_dir = os.path.join("MLtasks", "data", "uci_wine_quality")
    csv_path = _download_and_extract_wine_red(data_dir)
    X, y = _load_wine_red_csv(csv_path)

    n = X.shape[0]
    idx = np.random.permutation(n)
    n_train = int(train_ratio * n)
    tr, va = idx[:n_train], idx[n_train:]

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    # Standardize using TRAIN stats
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sigma
    X_va = (X_va - mu) / sigma

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {"train_mu": mu, "train_sigma": sigma, "csv_path": csv_path}


def build_model(device=None):
    model = torch.nn.Linear(get_task_metadata()["input_dim"], 1)
    if device is not None:
        model.to(device)
    return model


def train(model, train_loader, val_loader, device,
          lr: float = 0.05, momentum: float = 0.9, weight_decay: float = 1e-3,
          epochs: int = 500):
    """
    New features:
    - SGD with momentum
    - StepLR scheduler
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        sched.step()

        if epoch % 100 == 0 or epoch == 1:
            tr = evaluate(model, train_loader, device)
            va = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:4d} | train_mse={tr['mse']:.4f} val_mse={va['mse']:.4f} lr={sched.get_last_lr()[0]:.5f}")

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
    print("Loaded from:", extras["csv_path"])

    model = build_model(device=device)
    model = train(model, train_loader, val_loader, device=device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print("TRAIN:", train_m)
    print("VAL  :", val_m)

    # Reasonable thresholds for simple linear model on wine-quality
    exit_code = 0
    try:
        assert val_m["r2"] > 0.20, f"Val R2 too low: {val_m['r2']:.4f}"
        assert val_m["mse"] < 0.80, f"Val MSE too high: {val_m['mse']:.4f}"
    except AssertionError as e:
        print("FAIL:", str(e))
        exit_code = 1

    save_artifacts(model, {"train": train_m, "val": val_m}, out_dir)
    sys.exit(exit_code)