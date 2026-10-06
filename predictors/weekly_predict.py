#!/usr/bin/env python3
"""AI Weekly Stock Prediction v2 — GitHub Actions Automation.

Predicts next week's direction (5 trading days) for US stocks and indices.
Results stored in data/weekly_predictions.csv (committed to repo automatically).

Settings (tickers, data source, features, XGB params, ...) live in
config/weekly.toml merged over config/common.toml.
Shared logic (data, model, CSV log, trading, reporting) lives in common/.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

from common import indicators as ind
from common.config import load_config
from common.data import load_macro_closes
from common.pipeline import PredictionJob, run

CONFIG = load_config("weekly")

# ════════════════════════════════════════════════════════════════
# MACRO DATA
# ════════════════════════════════════════════════════════════════


def get_macro_data(years):
    """Download macro series and derive the weekly-scale macro features.

    Args:
        years (int): Years of macro history to download.

    Returns:
        pandas.DataFrame: Macro closes plus ``VIX_SMA20``, ``VIX_weekly_chg``,
        ``SPY_return_5d``, ``SPY_return_20d``, ``SPY_vs_SMA50``,
        ``Bond_return_5d``, ``Dollar_return_5d`` and ``Gold_return_5d``.

    Raises:
        RuntimeError: If required macro series cannot be downloaded.
    """
    macro = load_macro_closes(years, CONFIG["macro"]["symbols"])

    # VIX
    macro["VIX_SMA20"] = macro["VIX"].rolling(20).mean()
    macro["VIX_weekly_chg"] = macro["VIX"].pct_change(5)

    # SPY market regime
    macro["SPY_return_5d"] = macro["SPY"].pct_change(5)
    macro["SPY_return_20d"] = macro["SPY"].pct_change(20)
    macro["SPY_vs_SMA50"] = macro["SPY"] / macro["SPY"].rolling(50).mean()

    # Cross-asset weekly returns (5d scale matches HORIZON)
    macro["Bond_return_5d"] = macro["TLT"].pct_change(5)
    macro["Dollar_return_5d"] = macro["UUP"].pct_change(5)
    macro["Gold_return_5d"] = macro["GLD"].pct_change(5)

    return macro


# ════════════════════════════════════════════════════════════════
# 31 FEATURES — TUNED FOR WEEKLY PREDICTION
#
# Key differences from daily:
# - No ultra-short features (overnight gap, 2d/3d returns, intraday range)
# - Wider lookback windows (20d/60d vol, 20d close position, 50/200 SMA)
# - Cross-asset uses 5d returns (weekly scale) not 1d
# - Weekly VIX change instead of daily
# - Last week's return as lagged context
# - No stochastic (too fast) — RSI + MACD are better at weekly
# ════════════════════════════════════════════════════════════════


def create_features(df, macro_df):
    """Build the weekly feature set for one ticker.

    Args:
        df (pandas.DataFrame): Daily OHLCV price data indexed by date.
        macro_df (pandas.DataFrame): Output of :func:`get_macro_data`.

    Returns:
        pandas.DataFrame: Features indexed like ``df``; includes every column
        in the configured ``feature_cols`` (early rows are NaN while rolling windows fill).
    """
    feat = pd.DataFrame(index=df.index)
    _, high, low, close, volume = ind.ohlcv(df)

    # ── Momentum (5) — weekly-appropriate timescales ──
    for n in [5, 10, 20, 60]:
        feat[f"Return_{n}d"] = close.pct_change(n)
    feat["Return_5d_lag1"] = feat["Return_5d"].shift(5)  # last week's return

    # ── Trend (3) — includes longer SMA pair for weekly ──
    feat["SMA_10_50_ratio"] = close.rolling(10).mean() / close.rolling(50).mean()
    feat["SMA_50_200_ratio"] = close.rolling(50).mean() / close.rolling(200).mean()
    feat["Price_vs_SMA200"] = close / close.rolling(200).mean()

    # ── Volatility (4) — wider windows for weekly noise ──
    daily_ret = close.pct_change()
    feat["Volatility_20d"] = daily_ret.rolling(20).std()
    feat["Volatility_60d"] = daily_ret.rolling(60).std()
    feat["Vol_ratio_20_60"] = feat["Volatility_20d"] / feat["Volatility_60d"]
    feat["ATR_14"] = ind.atr_pct(high, low, close, 14)

    # ── Price Position (2) — 20-day range for weekly context ──
    feat["Close_pos_20d"] = ind.close_position(close, high, low, 20)
    feat["BB_pct"] = ind.bollinger_pct(close, 20)

    # ── Volume (3) ──
    feat["Volume_ratio"] = volume / volume.rolling(20).mean()
    feat["Up_volume_ratio"] = ind.up_volume_ratio(close, volume, 20)
    feat["OBV_slope"] = ind.obv_slope(close, volume, 20)

    # ── Technical (3) — RSI and MACD are effective at weekly ──
    feat["RSI_14"] = ind.rsi(close, 14)
    macd_hist = ind.macd_hist(close)
    feat["MACD_hist"] = macd_hist
    feat["RSI_5d_change"] = feat["RSI_14"] - feat["RSI_14"].shift(5)

    # ── VIX (3) — fear gauge with weekly change ──
    feat = ind.add_vix_features(feat, macro_df, extra_cols=["VIX_weekly_chg"])

    # ── Market Regime (3) — broad market weekly context ──
    feat = ind.join_macro(feat, macro_df, ["SPY_return_5d", "SPY_return_20d", "SPY_vs_SMA50"])

    # ── Cross-Asset (3) — weekly returns match HORIZON ──
    feat = ind.join_macro(feat, macro_df, ["Bond_return_5d", "Dollar_return_5d", "Gold_return_5d"])

    # ── Momentum Dynamics (2) — weekly-scale rate of change ──
    feat["MACD_accel_5d"] = macd_hist - macd_hist.shift(5)
    feat["Vol_accel"] = feat["Volatility_20d"] - feat["Volatility_20d"].shift(5)

    return feat


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    """Run the weekly prediction job with config/weekly.toml."""
    run(PredictionJob.from_config(CONFIG, create_features, get_macro_data))


if __name__ == "__main__":
    main()
