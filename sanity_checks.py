"""Leakage sanity checks -- run BEFORE trusting the real model.

Check 1 (direct leakage / wiring): put the binary target inside X. After
    just 1 epoch the model should reach ~100% accuracy. This only confirms
    the training pipeline is wired correctly (a model CAN learn a label
    that is literally handed to it).

Check 2 (indirect leakage): with the real features the model should NOT
    beat chance -- accuracy stays ~0.5, i.e. the current-day-indicator
    features do not leak the future. Then we deliberately shift next-day
    close into the window and train ~50 epochs; accuracy should jump high,
    proving this harness would catch leakage if it were present.

Run:  python sanity_checks.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from data import download_data
from features import compute_features
from windowing import split_by_date
from model import CNNLSTM

# Keep the checks fast: a few tickers and a capped training set are plenty
# to reveal leakage. Raise these if you want a stricter test.
SANITY_TICKERS = config.TICKERS[:4]
MAX_TRAIN_SAMPLES = 6000
SEED = 0


def _windows_for_ticker(df, window_length, inject=None):
    """Windows for one ticker, optionally appending a leaked feature column.

    inject: None, or a function df -> Series (aligned to df.index) whose
    values are appended as an extra feature at every timestep. Used to
    simulate leakage.
    """
    feats = compute_features(df)
    close = df["Close"].reindex(feats.index)
    next_ret = close.shift(-1) / close - 1
    y_bin = (next_ret > 0).astype(int)

    if inject is not None:
        feats = feats.assign(_LEAK_=inject(df).reindex(feats.index))

    values = feats.to_numpy(dtype=np.float32)
    X, y, dates = [], [], []
    for i in range(window_length - 1, len(feats)):
        if np.isnan(next_ret.iloc[i]):          # no next-day close -> skip
            continue
        window = values[i - window_length + 1: i + 1]
        if np.isnan(window).any():              # guard injected NaNs
            continue
        X.append(window)
        y.append(int(y_bin.iloc[i]))
        dates.append(feats.index[i])
    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(dates, "datetime64[ns]"))


def _build(inject=None):
    data = download_data(SANITY_TICKERS)
    Xs, ys, ds = [], [], []
    for ticker in SANITY_TICKERS:
        X, y, d = _windows_for_ticker(data[ticker], config.WINDOW_LENGTH, inject)
        Xs.append(X)
        ys.append(y)
        ds.append(d)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)


@torch.no_grad()
def _accuracy(model, X, y, device, batch_size=512):
    if len(X) == 0:
        return float("nan")
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        preds.append((torch.sigmoid(model(xb).squeeze(-1)) >= 0.5).cpu().numpy())
    return float((np.concatenate(preds) == y).mean())


def _train_and_eval(X, y, dates, epochs):
    """Train a fresh CNN-LSTM and return (train_acc, test_acc).

    Dropout is disabled here: a leakage probe wants the model to memorize
    any available signal as hard as possible, so we don't want the real
    model's regularization fighting it.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Time split; raw arg is unused here, scaler is not saved (path=None).
    split = split_by_date(X, y, y.astype(float), dates)
    Xtr, ytr = split["X_train"], split["y_train"]

    if len(Xtr) > MAX_TRAIN_SAMPLES:            # cap for speed
        idx = np.random.choice(len(Xtr), MAX_TRAIN_SAMPLES, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    model = CNNLSTM(X.shape[2], dropout=0.0).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # Slightly higher lr than training so an injected leak is memorized fast
    # (Check 1 must reach ~100% in a single epoch).
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr).float(),
                      torch.from_numpy(ytr.astype(np.float32))),
        batch_size=128, shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    return (_accuracy(model, Xtr, ytr, device),
            _accuracy(model, split["X_test"], split["y_test"], device))


def check1_direct_leakage():
    print("\n=== Check 1: direct leakage (binary target left inside X) ===")
    inject = lambda df: (df["Close"].shift(-1) > df["Close"]).astype(float)
    X, y, dates = _build(inject=inject)
    train_acc, test_acc = _train_and_eval(X, y, dates, epochs=1)
    print(f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")
    ok = test_acc >= 0.90
    print(f"[{'PASS' if ok else 'FAIL'}] expected ~1.0 -- the pipeline can learn a label handed to it in X")
    return ok


def check2_indirect_leakage():
    print("\n=== Check 2: indirect leakage ===")

    # 2a) Real features only -- must NOT beat chance.
    X, y, dates = _build(inject=None)
    _, test_acc_clean = _train_and_eval(X, y, dates, epochs=30)
    ok_a = test_acc_clean <= 0.60
    print(f"(2a) real features    test_acc={test_acc_clean:.3f}")
    print(f"     [{'PASS' if ok_a else 'FAIL'}] expected ~0.5 -- current-day-indicator features do NOT leak")

    # 2b) Shift next-day close into the window -- must become predictable.
    # Expressed relative to today's close so the leak is scale-free across
    # tickers (a raw price column would be swamped by cross-ticker scale).
    inject = lambda df: df["Close"].shift(-1) / df["Close"]
    Xl, yl, dl = _build(inject=inject)
    _, test_acc_leak = _train_and_eval(Xl, yl, dl, epochs=50)
    ok_b = test_acc_leak >= 0.85
    print(f"(2b) + next-day close test_acc={test_acc_leak:.3f}")
    print(f"     [{'PASS' if ok_b else 'FAIL'}] expected high -- harness detects future data in the window")

    return ok_a and ok_b


if __name__ == "__main__":
    r1 = check1_direct_leakage()
    r2 = check2_indirect_leakage()
    print("\n=== SUMMARY ===")
    print(f"Check 1 (direct leakage / wiring): {'PASS' if r1 else 'FAIL'}")
    print(f"Check 2 (indirect leakage)       : {'PASS' if r2 else 'FAIL'}")
