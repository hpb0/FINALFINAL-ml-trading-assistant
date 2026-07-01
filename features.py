"""Feature engineering for a single ticker's OHLCV DataFrame.

LEAKAGE RULE (critical):
    Every feature on a given row uses only data available at or before
    that day. Nothing here reads future values -- the next-day target is
    built separately in windowing.py via Close.shift(-1). The raw price
    *level* we feed in is YesterdayClose (a lagged value), not the naked
    current close, and the "Yesterday*" log-return features are prior-day
    returns (shifted by 1). Technical indicators are evaluated as of the
    current decision day, whose close is known when we forecast the next
    day.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def compute_features(df):
    """Return a DataFrame of features aligned to df's index (NaN rows dropped)."""
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Daily close log return, reused below.
    logr = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df.index)

    # --- Lagged (prior-day) price / volume features ---
    out["YesterdayClose"] = close.shift(1)
    out["YesterdayOpenLogR"] = np.log(open_ / open_.shift(1)).shift(1)
    out["YesterdayHighLogR"] = np.log(high / high.shift(1)).shift(1)
    out["YesterdayLowLogR"] = np.log(low / low.shift(1)).shift(1)
    out["YesterdayVolumeLogR"] = np.log(volume / volume.shift(1)).shift(1)
    out["YesterdayCloseLogR"] = logr.shift(1)

    # --- Simple moving averages ---
    out["MA10"] = close.rolling(10).mean()
    out["MA20"] = close.rolling(20).mean()
    out["MA30"] = close.rolling(30).mean()

    # --- Calendar features ---
    out["DayOfWeek"] = df.index.dayofweek
    out["DayOfMonth"] = df.index.day
    out["MonthNumber"] = df.index.month

    # --- Exponential moving averages ---
    out["EMA10"] = EMAIndicator(close, window=10).ema_indicator()
    out["EMA30"] = EMAIndicator(close, window=30).ema_indicator()

    # --- Momentum / trend indicators (ta library) ---
    out["RSI"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)  # defaults: fast=12, slow=26, signal=9
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    bb = BollingerBands(close, window=20, window_dev=2)
    out["BollingerUpper"] = bb.bollinger_hband()
    out["BollingerLower"] = bb.bollinger_lband()

    # --- Realised volatility of daily log returns ---
    out["Volatility_10"] = logr.rolling(10).std()
    out["Volatility_20"] = logr.rolling(20).std()
    out["Volatility_30"] = logr.rolling(30).std()

    # --- On-balance volume ---
    out["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # --- Z-score of close vs its 20-day mean/std ---
    out["ZScore"] = (close - close.rolling(20).mean()) / close.rolling(20).std()

    # --- STUB alt-data placeholders ---
    # 0.0 stand-ins for real insider-trade / sentiment / news feeds.
    # Wire real data in here later; kept as columns so the model shape is
    # already correct.
    out["insider_shares"] = 0.0
    out["insider_amount"] = 0.0
    out["insider_buy_flag"] = 0.0
    out["sentiment"] = 0.0
    out["num_articles"] = 0.0

    # --- Extra engineered features ---
    out["overnight_gap"] = np.log(open_ / close.shift(1))
    out["abnormal_vol"] = volume / volume.rolling(20).mean()
    out["volatility_5d"] = logr.rolling(5).std()
    out["volatility_20d"] = logr.rolling(20).std()
    out["momentum_5d"] = close / close.shift(5) - 1
    out["momentum_20d"] = close / close.shift(20) - 1
    out["skew_5d"] = logr.rolling(5).skew()
    out["intraday_range"] = (high - low) / close
    out["sentiment_change"] = 0.0  # STUB placeholder

    # Drop warmup rows and any inf produced by zero-volume divisions.
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


if __name__ == "__main__":
    # Quick verification: compute features for one ticker and print shape.
    from data import download_data

    ticker = "AAPL"
    df = download_data([ticker])[ticker]
    feats = compute_features(df)
    print(f"{ticker}: raw {df.shape} -> features {feats.shape}")
    print(f"n_features = {feats.shape[1]}")
    print("columns:", list(feats.columns))
    print(feats.tail(3))
