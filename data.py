"""Download and cache OHLCV price data.

We use yfinance for daily OHLCV per ticker and cache each ticker to a
parquet file so we don't re-download on every run.

This project is also *conceptually* built to consume alt-data such as
insider trades, sentiment and news volume. Those feeds are not wired up
here; they are stubbed as 0.0 placeholder features in features.py.
"""

import os

import pandas as pd
import yfinance as yf

import config

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker):
    return os.path.join(config.DATA_CACHE_DIR, f"{ticker}.parquet")


def download_data(tickers=None, start=None, end=None, force=False):
    """Return {ticker: DataFrame} of daily OHLCV, using the parquet cache.

    Each DataFrame is indexed by date with columns
    [Open, High, Low, Close, Volume]. Downloads only tickers not already
    cached (pass force=True to re-download everything).
    """
    tickers = tickers or config.TICKERS
    start = start or config.START_DATE
    end = end or config.END_DATE

    os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)

    data = {}
    for ticker in tickers:
        path = _cache_path(ticker)
        if os.path.exists(path) and not force:
            df = pd.read_parquet(path)
        else:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            # yfinance may return MultiIndex columns; flatten to plain names.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[OHLCV_COLUMNS].dropna()
            df.to_parquet(path)
        data[ticker] = df
    return data


if __name__ == "__main__":
    # Quick verification: download (or load from cache) and print shapes.
    data = download_data()
    for ticker, df in data.items():
        print(f"{ticker:6s} {df.shape}  {df.index.min().date()} -> {df.index.max().date()}")
