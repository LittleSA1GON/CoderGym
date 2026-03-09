import os
import sys
import zipfile
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

TASK_ID = os.path.basename(os.path.dirname(__file__))

# UCI lists data_banknote_authentication.txt in dataset files. :contentReference[oaicite:9]{index=9}
UCI_BANKNOTE_ZIP_URL = "https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip"


def get_task_metadata():
    return {
        "task_type": "binary_classification",
        "model_type": "logistic_regression",
        "input_dim": 4,
        "output_dim": 1,
        "description": "Logistic regression on UCI Banknote Authentication using LBFGS + L2 penalty.",
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


def _download_and_extract_banknote(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "banknote.zip")
    txt_path = os.path.join(data_dir, "data_banknote_authentication.txt")

    if os.path.exists(txt_path):
        return txt_path

    if not os.path.exists(zip_path):
        print("Downloading UCI Banknote Authentication zip...")
        urllib.request.urlretrieve(UCI_BANKNOTE_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains data_banknote_authentication.txt (as listed by UCI). :contentReference[oaicite:10]{index=10}
        zf.extract("data_banknote_authentication.txt", path=data_dir)

    return txt_path


def make_dataloaders(batch_size: int = 512, train_ratio: float = 0.8):
    """
    Newbie-friendly:
    - Download from UCI once and cache
    - Parse CSV-like txt: 4 features + class label
    - Standardize features using TRAIN stats
    """
    data_dir = os.path.join("MLtasks", "data", "uci_banknote")
    txt_path = _download_and_extract_banknote(data_dir)

    data = np.loadtxt(txt_path, delimiter=",", dtype=np.float32)
    X = data[:, :4]
    y = data[:, 4].reshape(-1, 1)  # 0/1

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

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, {"txt_path": txt_path, "train_mu": mu, "train_sigma": sigma}


def build_model(device=None):
    model = torch.nn.Linear(get_task_metadata()["input_dim"], 1)
    if device is not None:
        model.to(device)
    return model


def train(model, train_loader, val_loader, device, l2: float = 1e-3):
    """
    New feature:
    - Use LBFGS optimizer (requires a closure) :contentReference[oaicite:11]{index=11}
    - Full-batch training for simplicity
    """
    # Full-batch tensors
    Xs, ys = [], []
    for xb, yb in train_loader:
        Xs.append(xb)
        ys.append(yb)
    X_full = torch.cat(Xs, dim=0).to(device)
    y_full = torch.cat(ys, dim=0).to(device)

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = model(X_full)
        loss = F.binary_cross_entropy_with_logits(logits, y_full)

        # simple L2 penalty
        l2_term = 0.0
        for p in model.parameters():
            l2_term = l2_term + torch.sum(p * p)
        loss = loss + l2 * l2_term

        loss.backward()
        return loss

    opt.step(closure)
    return model


def evaluate(model, data_loader, device):
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            probs_all.append(torch.sigmoid(model(xb)))
            y_all.append(yb)

    y_prob = torch.cat(probs_all, dim=0)
    y_true = torch.cat(y_all, dim=0)
    y_pred = (y_prob >= 0.5).float()

    acc = torch.mean((y_pred == y_true).float()).item()
    logloss = F.binary_cross_entropy(y_prob, y_true).item()

    # protocol-like extras
    mse = torch.mean((y_true - y_prob) ** 2).item()
    r2 = _r2(y_true, y_prob)

    return {"accuracy": acc, "logloss": logloss, "mse": mse, "r2": r2}


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
    print("Loaded from:", extras["txt_path"])

    model = build_model(device=device)
    model = train(model, train_loader, val_loader, device=device, l2=1e-3)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print("TRAIN:", train_m)
    print("VAL  :", val_m)

    exit_code = 0
    try:
        assert val_m["accuracy"] > 0.97, f"Val accuracy too low: {val_m['accuracy']:.3f}"
        assert val_m["logloss"] < 0.20, f"Val logloss too high: {val_m['logloss']:.3f}"
    except AssertionError as e:
        print("FAIL:", str(e))
        exit_code = 1

    save_artifacts(model, {"train": train_m, "val": val_m}, out_dir)
    sys.exit(exit_code)