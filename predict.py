"""Generate Buy/Hold/Sell signals for a date using the trained model.

predict_signals() loads the saved CNN-LSTM + scaler, builds the feature
window ending on the requested date for each ticker (reusing features.py
and the same lagged, leakage-safe features used in training), runs
Monte-Carlo dropout for P(up) + uncertainty, and turns that into a signal
plus a derived predicted price.
"""

import numpy as np
import pandas as pd
import torch
import joblib

import config
from data import download_data
from features import compute_features
from model import CNNLSTM

# Monte-Carlo dropout passes used for P(up) + uncertainty.
MC_SAMPLES = 50


def load_model(model_path=None, scaler_path=None):
    """Load the trained CNN-LSTM and its fitted StandardScaler."""
    model_path = model_path or config.MODEL_PATH
    scaler_path = scaler_path or config.SCALER_PATH
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = CNNLSTM(ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def _recent_volatility(close, pos, lookback=20):
    """Std of daily simple returns over the `lookback` days ending at `pos`.

    Used only to scale the derived predicted price; uses no future data.
    """
    window = close.iloc[max(0, pos - lookback + 1): pos + 1]
    vol = window.pct_change().std()
    return float(vol) if np.isfinite(vol) else 0.0


def predict_signals(date=None, held=None, model=None, scaler=None):
    """Return a list of signal dicts for `date` (default: latest available).

    Each dict: {ticker, current_price, predicted_price, prob_up,
    uncertainty, signal}.

    held: iterable of tickers currently held -- Sell only fires for a held
    name. Defaults to empty, so outside the backtest Sell never triggers
    (no shorting; Sell is an exit only).
    """
    held = set(held or [])
    if model is None or scaler is None:
        model, scaler = load_model()

    data = download_data(config.TICKERS)
    target = pd.Timestamp(date) if date else None

    results = []
    for ticker in config.TICKERS:
        df = data[ticker]
        feats = compute_features(df)
        close = df["Close"].reindex(feats.index)

        # Position of the last feature row on or before the target date
        # (or the most recent row when no date is given).
        if target is None:
            i = len(feats) - 1
        else:
            on_or_before = feats.index[feats.index <= target]
            if len(on_or_before) == 0:
                continue
            i = feats.index.get_loc(on_or_before[-1])

        if i < config.WINDOW_LENGTH - 1:
            continue  # not enough history for a full window

        window = feats.iloc[i - config.WINDOW_LENGTH + 1: i + 1].to_numpy(np.float32)
        window = scaler.transform(window).astype(np.float32)
        x = torch.from_numpy(window).unsqueeze(0)          # (1, W, F)

        mean, std = model.mc_predict(x, n_samples=MC_SAMPLES)
        prob_up = float(mean.item())
        uncertainty = float(std.item())

        current_price = float(close.iloc[i])
        # Derived predicted price -- NOT a precise forecast. The model only
        # outputs P(up), so we translate it into an expected next-day return
        # of (2*P(up) - 1) scaled by recent daily volatility, then apply it
        # to the current close. Purely an illustrative estimate.
        expected_return = (2 * prob_up - 1) * _recent_volatility(close, i)
        predicted_price = current_price * (1 + expected_return)

        if prob_up >= config.BUY_THRESHOLD:
            signal = "Buy"
        elif prob_up <= config.SELL_THRESHOLD and ticker in held:
            signal = "Sell"
        else:
            signal = "Hold"

        results.append({
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "prob_up": round(prob_up, 4),
            "uncertainty": round(uncertainty, 4),
            "signal": signal,
        })
    return results


if __name__ == "__main__":
    signals = predict_signals()
    header = f"{'Ticker':<8}{'Price':>10}{'Pred':>10}{'P(up)':>8}{'Uncert':>9}   Signal"
    print(header)
    print("-" * len(header))
    for s in signals:
        print(f"{s['ticker']:<8}{s['current_price']:>10.2f}{s['predicted_price']:>10.2f}"
              f"{s['prob_up']:>8.3f}{s['uncertainty']:>9.3f}   {s['signal']}")
