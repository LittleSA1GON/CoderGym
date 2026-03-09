import os
import sys
import zipfile
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

TASK_ID = os.path.basename(os.path.dirname(__file__))

# UCI zip URL revealed by the dataset's download link. :contentReference[oaicite:4]{index=4}
UCI_WDBC_ZIP_URL = "https://archive.ics.uci.edu/static/public/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip"


def get_task_metadata():
    return {
        "task_type": "binary_classification",
        "model_type": "logistic_regression",
        "input_dim": 30,
        "output_dim": 1,
        "description": "Logistic regression on UCI WDBC using pos_weight + threshold tuning for F1 (no sklearn).",
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


def _extract_if_missing(zip_path: str, member_name: str, out_path: str):
    if os.path.exists(out_path):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(member_name, path=os.path.dirname(out_path))
    # ensure exact expected path
    extracted = os.path.join(os.path.dirname(out_path), member_name)
    if extracted != out_path and os.path.exists(extracted):
        os.replace(extracted, out_path)


def _load_wdbc_data(wdbc_path: str):
    """
    Each row: ID, Diagnosis (M/B), then 30 float features.
    UCI confirms fields and that wdbc.data is the dataset file. :contentReference[oaicite:5]{index=5}
    """
    X_list, y_list = [], []
    with open(wdbc_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 32:
                continue
            diagnosis = parts[1].strip()
            feats = [float(v) for v in parts[2:]]  # 30 floats

            # define positive class = malignant ('M')
            y = 1.0 if diagnosis == "M" else 0.0

            X_list.append(feats)
            y_list.append(y)

    X = np.array(X_list, dtype=np.float32)                 # [N,30]
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)  # [N,1]
    return X, y


def _binary_metrics_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float):
    y_true = y_true.view(-1)
    y_prob = y_prob.view(-1)
    y_pred = (y_prob >= threshold).float()

    acc = torch.mean((y_pred == y_true).float()).item()
    tp = torch.sum((y_pred == 1) & (y_true == 1)).float()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).float()

    precision = (tp / (tp + fp + 1e-12)).item()
    recall = (tp / (tp + fn + 1e-12)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-12)) if (precision + recall) > 0 else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": float(f1)}


def make_dataloaders(batch_size: int = 64, train_ratio: float = 0.8):
    """
    - Download UCI zip
    - Extract wdbc.data
    - Train/val split
    - Standardize using TRAIN stats only
    - Compute pos_weight from TRAIN
    """
    data_dir = os.path.join("MLtasks", "data", "uci_wdbc")
    zip_path = os.path.join(data_dir, "wdbc.zip")
    wdbc_path = os.path.join(data_dir, "wdbc.data")

    _download_if_missing(UCI_WDBC_ZIP_URL, zip_path)
    _extract_if_missing(zip_path, "wdbc.data", wdbc_path)

    X, y = _load_wdbc_data(wdbc_path)
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

    # pos_weight = (#neg / #pos) using TRAIN only
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / (pos + 1e-12)], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {"zip_path": zip_path, "wdbc_path": wdbc_path, "pos_weight": pos_weight}


def build_model(device=None):
    model = torch.nn.Linear(get_task_metadata()["input_dim"], 1)
    if device is not None:
        model.to(device)
    return model


def train(model, train_loader, val_loader, device, pos_weight: torch.Tensor,
          lr: float = 0.02, epochs: int = 400):
    """
    Feature additions:
    - BCEWithLogitsLoss(pos_weight=...)
    """
    pos_weight = pos_weight.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if epoch % 100 == 0 or epoch == 1:
            tr = evaluate(model, train_loader, device, threshold=0.5)
            va = evaluate(model, val_loader, device, threshold=0.5)
            print(f"Epoch {epoch:4d} | train_acc={tr['accuracy']:.3f} val_acc={va['accuracy']:.3f} val_f1={va['f1']:.3f}")

    return model


def _collect_probs_and_labels(model, loader, device):
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            probs_all.append(torch.sigmoid(model(xb)))
            y_all.append(yb)
    return torch.cat(probs_all, dim=0), torch.cat(y_all, dim=0)


def _tune_threshold_on_train(model, train_loader, device):
    y_prob, y_true = _collect_probs_and_labels(model, train_loader, device)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        m = _binary_metrics_from_probs(y_true, y_prob, float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t, best_f1


def evaluate(model, data_loader, device, threshold: float = 0.5):
    y_prob, y_true = _collect_probs_and_labels(model, data_loader, device)

    mse = torch.mean((y_true - y_prob) ** 2).item()
    r2 = _r2(y_true, y_prob)
    logloss = F.binary_cross_entropy(y_prob, y_true).item()

    m = _binary_metrics_from_probs(y_true, y_prob, threshold)
    m.update({"mse": mse, "r2": r2, "logloss": logloss, "threshold": threshold})
    return m


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(model(X.to(device))).cpu().numpy()


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
    print("Zip:", extras["zip_path"])
    print("Data:", extras["wdbc_path"])
    print("pos_weight:", float(extras["pos_weight"].item()))

    model = build_model(device=device)
    model = train(model, train_loader, val_loader, device=device, pos_weight=extras["pos_weight"])

    # threshold=0.5 baseline
    train_05 = evaluate(model, train_loader, device, threshold=0.5)
    val_05 = evaluate(model, val_loader, device, threshold=0.5)

    # tune threshold on TRAIN for best F1
    best_t, best_train_f1 = _tune_threshold_on_train(model, train_loader, device)
    train_best = evaluate(model, train_loader, device, threshold=best_t)
    val_best = evaluate(model, val_loader, device, threshold=best_t)

    print("TRAIN (t=0.5):", train_05)
    print("VAL   (t=0.5):", val_05)
    print(f"Best threshold on TRAIN: t={best_t:.2f} (train_f1={best_train_f1:.3f})")
    print("TRAIN (t=best):", train_best)
    print("VAL   (t=best):", val_best)

    exit_code = 0
    try:
        assert val_best["accuracy"] > 0.92, f"Val accuracy too low: {val_best['accuracy']:.3f}"
        assert val_best["f1"] > 0.92, f"Val F1 too low: {val_best['f1']:.3f}"
    except AssertionError as e:
        print("FAIL:", str(e))
        exit_code = 1

    save_artifacts(
        model,
        {"train_t05": train_05, "val_t05": val_05, "train_best": train_best, "val_best": val_best},
        out_dir,
    )
    sys.exit(exit_code)