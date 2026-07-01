"""Train the CNN-LSTM once, then save the model, scaler and training curves.

Run order: this is step 1 (after installing deps). It downloads data,
builds features + windows, does the time split, trains for config.EPOCHS
and writes models/cnn_lstm.pt, models/scaler.joblib and
models/training_curves.png.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")            # headless: just write a PNG
import matplotlib.pyplot as plt

import config
from data import download_data
from windowing import make_windows, split_by_date
from model import CNNLSTM

BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def build_dataset(tickers=None, window_length=None):
    """Window every ticker and concatenate into one big dataset."""
    tickers = tickers or config.TICKERS
    window_length = window_length or config.WINDOW_LENGTH
    data = download_data(tickers)

    Xs, ys, raws, dates = [], [], [], []
    for ticker in tickers:
        X, y, raw, d = make_windows(data[ticker], window_length)
        Xs.append(X)
        ys.append(y)
        raws.append(raw)
        dates.append(d)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(raws), np.concatenate(dates))


@torch.no_grad()
def accuracy(model, X, y, device, batch_size=512):
    """Direction accuracy (dropout off) over an array, batched."""
    if len(X) == 0:
        return float("nan")
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        logits = model(xb).squeeze(-1)
        preds.append((torch.sigmoid(logits) >= 0.5).cpu().numpy())
    return float((np.concatenate(preds) == y).mean())


def train(early_stopping=False, patience=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    X, y, raw, dates = build_dataset()
    split = split_by_date(X, y, raw, dates, scaler_path=config.SCALER_PATH)
    print(f"train {split['X_train'].shape}  val {split['X_val'].shape}  test {split['X_test'].shape}")

    n_features = X.shape[2]
    model = CNNLSTM(n_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    Xtr = torch.from_numpy(split["X_train"]).float()
    ytr = torch.from_numpy(split["y_train"].astype(np.float32))
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_acc": [], "val_acc": []}
    best_val, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_acc = accuracy(model, split["X_train"], split["y_train"], device)
        val_acc = accuracy(model, split["X_val"], split["y_val"], device)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:3d}/{config.EPOCHS}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

        if early_stopping:
            if val_acc > best_val:
                best_val, bad_epochs = val_acc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stopping at epoch {epoch} (best val_acc {best_val:.3f})")
                    break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    # Save model (with n_features so predict.py can rebuild it) + curves.
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": n_features}, config.MODEL_PATH)
    print(f"saved model  -> {config.MODEL_PATH}")
    print(f"saved scaler -> {config.SCALER_PATH}")

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN-LSTM training")
    plt.legend()
    curve_path = os.path.join(config.MODELS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=120)
    print(f"saved curves -> {curve_path}")

    return model, history


if __name__ == "__main__":
    # Optional early stopping: set early_stopping=True to enable (default off).
    train(early_stopping=False)
