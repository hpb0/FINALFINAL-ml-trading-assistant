"""Backtest the Top-3 strategy over the test period, with benchmarks.

run_backtest() walks the test period (prediction day > VAL_END) tracking a
portfolio. Each day it scores every ticker with the model, holds the 3
highest-P(up) Buy-signalled names, and marks them to the NEXT day's
realized return. It compares against buy-and-hold S&P 500 and a Monte
Carlo random baseline, and saves two plots to results/.

Leakage discipline: P(up) for a given day uses only the lagged features
available up to that day; the day's holdings are marked to the next day's
realized return, so no future price enters the decision.
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")            # headless: just write PNGs
import matplotlib.pyplot as plt

import config
from data import download_data
from windowing import make_windows
from predict import load_model, MC_SAMPLES

RESULTS_DIR = os.path.join(config.BASE_DIR, "results")
N_RANDOM_RUNS = 2000
TOP_K = 3


def _test_probs_and_returns(model, scaler):
    """Return (P, R): DataFrames [dates x tickers] of P(up) and realized
    next-day returns over the test period. Batches the model per ticker so
    we score every test day in one pass rather than day-by-day."""
    data = download_data(config.TICKERS)
    val_end = pd.Timestamp(config.VAL_END)
    prob_cols, ret_cols = {}, {}

    for ticker in config.TICKERS:
        X, y, raw, dates = make_windows(data[ticker], config.WINDOW_LENGTH)
        dates = pd.to_datetime(dates)
        mask = dates > val_end
        if mask.sum() == 0:
            continue

        Xt = X[mask]
        n_features = Xt.shape[2]
        Xs = scaler.transform(Xt.reshape(-1, n_features)).reshape(Xt.shape).astype(np.float32)
        mean, _ = model.mc_predict(torch.from_numpy(Xs), n_samples=MC_SAMPLES)

        idx = dates[mask]
        prob_cols[ticker] = pd.Series(mean.squeeze(-1).numpy(), index=idx)
        ret_cols[ticker] = pd.Series(raw[mask], index=idx)

    P = pd.DataFrame(prob_cols).sort_index()
    R = pd.DataFrame(ret_cols).sort_index()
    # Keep only days where every ticker has both a probability and a return.
    common = P.dropna().index.intersection(R.dropna().index)
    return P.loc[common], R.loc[common]


def _simulate_strategy(P, R):
    """Rebalance daily to the Top-3 Buy-signalled tickers and compound.

    Holding the Top-3 Buys implicitly implements Buy (a name entering the
    Top-3), Hold (staying in it) and Sell (dropping out, which includes any
    name falling to P(up) <= SELL_THRESHOLD). Exit only, no shorting.
    """
    values, holdings = [], []
    value = 1.0
    for d in P.index:
        probs_d = P.loc[d]
        buys = probs_d[probs_d >= config.BUY_THRESHOLD].sort_values(ascending=False)
        target = list(buys.index[:TOP_K])
        # Mark to the next day's realized return (R is already next-day return).
        day_ret = float(R.loc[d, target].mean()) if target else 0.0   # else all cash
        value *= (1.0 + day_ret)
        values.append(value)
        holdings.append(target)
    return np.array(values), holdings


def _random_baseline(R, n_runs=N_RANDOM_RUNS, seed=0):
    """Distribution of total returns from buying TOP_K random tickers/day."""
    rng = np.random.default_rng(seed)
    Rv = R.to_numpy()
    n_days, n_tickers = Rv.shape
    finals = np.ones(n_runs)
    for d in range(n_days):
        # TOP_K distinct random tickers for every run (argsort of noise).
        pick = rng.random((n_runs, n_tickers)).argsort(axis=1)[:, :TOP_K]
        finals *= (1.0 + Rv[d][pick].mean(axis=1))
    return finals - 1.0


def run_backtest(n_random=N_RANDOM_RUNS, save_plots=True):
    """Run the full backtest and return a results dict (JSON-friendly)."""
    model, scaler = load_model()
    P, R = _test_probs_and_returns(model, scaler)
    dates = P.index

    strat_values, _ = _simulate_strategy(P, R)
    strategy_return = float(strat_values[-1] - 1.0)

    # Buy-and-hold S&P 500 over the same window, normalized to start at 1.0.
    spx = download_data([config.SP500_TICKER])[config.SP500_TICKER]["Close"]
    spx = spx.reindex(dates, method="ffill")
    bh_values = (spx / spx.iloc[0]).to_numpy()
    buy_hold_return = float(bh_values[-1] - 1.0)

    # Monte Carlo random baseline.
    random_returns = _random_baseline(R, n_runs=n_random)
    random_mean = float(random_returns.mean())
    random_std = float(random_returns.std())
    sigma_above_random = (
        float((strategy_return - random_mean) / random_std) if random_std > 0 else 0.0
    )
    percentile = float((random_returns < strategy_return).mean() * 100)

    results = {
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "random_mean": random_mean,
        "random_std": random_std,
        "sigma_above_random": sigma_above_random,
        "percentile": percentile,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "strategy_values": strat_values.tolist(),
        "buy_hold_values": bh_values.tolist(),
        "random_returns": random_returns.tolist(),
    }

    if save_plots:
        _save_plots(results)
    return results


def _save_plots(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Histogram of random-baseline returns with the strategy marked.
    plt.figure()
    plt.hist(results["random_returns"], bins=50, color="lightgray", edgecolor="white")
    plt.axvline(results["strategy_return"], color="green", linewidth=2,
                label=f"Strategy {results['strategy_return'] * 100:.1f}%")
    plt.axvline(results["random_mean"], color="black", linestyle="--", linewidth=1,
                label=f"Random mean {results['random_mean'] * 100:.1f}%")
    plt.xlabel("total return over test period")
    plt.ylabel("frequency")
    plt.title("Top-3 strategy vs random baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "random_baseline_hist.png"), dpi=120)
    plt.close()

    # 2) Portfolio value vs buy-and-hold S&P 500.
    dates = pd.to_datetime(results["dates"])
    plt.figure()
    plt.plot(dates, results["strategy_values"], color="green", label="Top-3 strategy")
    plt.plot(dates, results["buy_hold_values"], color="gray", label="Buy & hold S&P 500")
    plt.xlabel("date")
    plt.ylabel("portfolio value (start = 1.0)")
    plt.title("Portfolio value over the test period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "portfolio_value.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    res = run_backtest()
    print("\n=== Backtest summary ===")
    print(f"period             : {res['start_date']} -> {res['end_date']}")
    print(f"strategy return    : {res['strategy_return'] * 100:7.2f}%")
    print(f"buy & hold S&P 500 : {res['buy_hold_return'] * 100:7.2f}%")
    print(f"random mean +/- std: {res['random_mean'] * 100:7.2f}% +/- {res['random_std'] * 100:.2f}%")
    print(f"sigma above random : {res['sigma_above_random']:.2f}")
    print(f"percentile         : {res['percentile']:.1f}%")
    print(f"plots saved to     : {RESULTS_DIR}")
