"""All project-wide constants live here."""

import os

# --- Tickers: ~15 liquid S&P 500 names ---
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "HD", "DIS",
]

# S&P 500 index, used as the buy-and-hold benchmark in the backtest.
SP500_TICKER = "^GSPC"

# --- Data date range (downloaded once, cached to parquet) ---
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# --- Three-way time split (by prediction-day date) ---
#   train:  START_DATE .. TRAIN_END
#   val:    TRAIN_END  .. VAL_END
#   test:   VAL_END    .. END_DATE
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"

# --- Windowing / model ---
WINDOW_LENGTH = 20        # trading days per input sequence
EPOCHS = 200

# --- Signal thresholds on P(up) ---
BUY_THRESHOLD = 0.7
SELL_THRESHOLD = 0.3

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_lstm.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
