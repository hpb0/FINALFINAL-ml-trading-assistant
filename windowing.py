"""Targets, sliding windows, and a time-based train/val/test split.

make_windows() turns one ticker's OHLCV into overlapping feature
sequences plus a next-day direction target. split_by_date() splits the
(optionally multi-ticker) windows by prediction-day date and fits the
StandardScaler on the training set only.
"""

import os

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

import config
from features import compute_features


def make_windows(df, window_length):
    """Build sliding windows for one ticker.

    Target: next-day return = Close.shift(-1) / Close - 1, labelled 1 if
    positive else 0. The raw next-day return is kept too so a predicted
    price can be derived later.

    Returns:
        X      float32 array (n_samples, window_length, n_features)
        y      int array   (n_samples,)  -- 1 if next day up else 0
        raw    float array (n_samples,)  -- next-day return (for price)
        dates  datetime64 array (n_samples,) -- prediction-day per sample
    """
    feats = compute_features(df)
    close = df["Close"].reindex(feats.index)

    next_ret = close.shift(-1) / close - 1        # return from day t to t+1
    y_bin = (next_ret > 0).astype(int)

    values = feats.to_numpy(dtype=np.float32)
    index = feats.index

    X, y, raw, dates = [], [], [], []
    for i in range(window_length - 1, len(feats)):
        # Window covers rows [i-window_length+1 .. i]; prediction day = i.
        # The last row has no next-day close, so its target is NaN -> skip.
        if np.isnan(next_ret.iloc[i]):
            continue
        X.append(values[i - window_length + 1: i + 1])
        y.append(int(y_bin.iloc[i]))
        raw.append(float(next_ret.iloc[i]))
        dates.append(index[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float64)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    return X, y, raw, dates


def split_by_date(X, y, raw, dates, train_end=None, val_end=None, scaler_path=None):
    """Split windows three ways by prediction-day date and scale features.

    A TIME split is used instead of a stock-percentage split because the
    latter lets the model memorize broad market regimes (e.g. the 2008
    crash) that appear in both train and test, producing falsely high
    backtest accuracy. Splitting by date keeps the future strictly unseen.

    The StandardScaler is fit on TRAIN ONLY, then applied to val and test.
    """
    train_end = pd.Timestamp(train_end or config.TRAIN_END)
    val_end = pd.Timestamp(val_end or config.VAL_END)
    dates = pd.to_datetime(dates)

    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end

    n_features = X.shape[2]
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, n_features))

    def scale(arr):
        if len(arr) == 0:
            return arr
        return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape).astype(np.float32)

    split = {
        "X_train": scale(X[train_mask]), "y_train": y[train_mask],
        "raw_train": raw[train_mask], "dates_train": dates[train_mask],
        "X_val": scale(X[val_mask]), "y_val": y[val_mask],
        "raw_val": raw[val_mask], "dates_val": dates[val_mask],
        "X_test": scale(X[test_mask]), "y_test": y[test_mask],
        "raw_test": raw[test_mask], "dates_test": dates[test_mask],
        "scaler": scaler,
    }

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return split


if __name__ == "__main__":
    # Quick verification: window one ticker and show the split sizes/shapes.
    from data import download_data

    ticker = "AAPL"
    df = download_data([ticker])[ticker]
    X, y, raw, dates = make_windows(df, config.WINDOW_LENGTH)
    print(f"{ticker}: X {X.shape}, y {y.shape}, raw {raw.shape}, dates {dates.shape}")
    print(f"date range: {dates.min()} -> {dates.max()}")
    print(f"positive-class rate: {y.mean():.3f}")

    split = split_by_date(X, y, raw, dates)
    for name in ("train", "val", "test"):
        print(f"{name:5s}: X {split['X_' + name].shape}, y {split['y_' + name].shape}")
