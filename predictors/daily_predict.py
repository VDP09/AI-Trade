#!/usr/bin/env python3
"""AI Daily Stock Prediction v2 — GitHub Actions Automation.

Results stored in data/predictions.csv (committed to repo automatically).
Dashboard rendered in GitHub Actions job summary.

Settings (tickers, data source, features, XGB params, ...) live in
config/daily.toml merged over config/common.toml.
Shared logic (data, model, CSV log, trading, reporting) lives in common/.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from common import indicators as ind
from common.config import load_config
from common.data import load_macro_closes
from common.pipeline import PredictionJob, run

CONFIG = load_config("daily")

# ════════════════════════════════════════════════════════════════
# MACRO DATA
# ════════════════════════════════════════════════════════════════


def get_macro_data(years):
    """Download macro series and derive the daily-scale macro features.

    Args:
        years (int): Years of macro history to download.

    Returns:
        pandas.DataFrame: Macro closes plus ``VIX_SMA20``, ``SPY_return_1d``,
        ``SPY_return_5d``, ``SPY_vs_SMA50``, ``Bond_return_1d``,
        ``Dollar_return_1d`` and ``Gold_return_1d``.

    Raises:
        RuntimeError: If required macro series cannot be downloaded.
    """
    macro = load_macro_closes(years, CONFIG["macro"]["symbols"])
    macro["VIX_SMA20"] = macro["VIX"].rolling(20).mean()
    macro["SPY_return_1d"] = macro["SPY"].pct_change(1)
    macro["SPY_return_5d"] = macro["SPY"].pct_change(5)
    macro["SPY_vs_SMA50"] = macro["SPY"] / macro["SPY"].rolling(50).mean()
    macro["Bond_return_1d"] = macro["TLT"].pct_change(1)
    macro["Dollar_return_1d"] = macro["UUP"].pct_change(1)
    macro["Gold_return_1d"] = macro["GLD"].pct_change(1)
    return macro


# ════════════════════════════════════════════════════════════════
# 38 FEATURE COLUMNS (37 conceptual features; day-of-week uses sin + cos)
# ════════════════════════════════════════════════════════════════


def create_features(df, macro_df):
    """Build the daily feature set for one ticker.

    Args:
        df (pandas.DataFrame): Daily OHLCV price data indexed by date.
        macro_df (pandas.DataFrame): Output of :func:`get_macro_data`.

    Returns:
        pandas.DataFrame: Features indexed like ``df``; includes every column
        in the configured ``feature_cols`` (early rows are NaN while rolling windows fill).
    """
    feat = pd.DataFrame(index=df.index)
    opn, high, low, close, volume = ind.ohlcv(df)

    # Momentum (6)
    for n in [1, 2, 3, 5, 20]:
        feat[f"Return_{n}d"] = close.pct_change(n)
    feat["Overnight_gap"] = opn / close.shift(1) - 1

    # Trend (2)
    feat["SMA_10_50_ratio"] = close.rolling(10).mean() / close.rolling(50).mean()
    feat["Price_vs_SMA200"] = close / close.rolling(200).mean()

    # Volatility (6)
    dr = close.pct_change()
    feat["Volatility_5d"] = dr.rolling(5).std()
    feat["Volatility_10d"] = dr.rolling(10).std()
    feat["Volatility_20d"] = dr.rolling(20).std()
    feat["Vol_ratio_5_20"] = feat["Volatility_5d"] / feat["Volatility_20d"]
    feat["ATR_14"] = ind.atr_pct(high, low, close, 14)
    feat["Intraday_range"] = (high - low) / close

    # Price Position (2)
    feat["Close_pos_5d"] = ind.close_position(close, high, low, 5)
    feat["BB_pct"] = ind.bollinger_pct(close, 20)

    # Volume (3)
    feat["Volume_ratio"] = volume / volume.rolling(20).mean()
    feat["Up_volume_ratio"] = ind.up_volume_ratio(close, volume, 10)
    feat["OBV_slope"] = ind.obv_slope(close, volume, 10)

    # Technical (3)
    feat["RSI_14"] = ind.rsi(close, 14)
    feat["Stochastic_K"] = (100 * ind.close_position(close, high, low, 5)).rolling(3).mean()
    macd_hist = ind.macd_hist(close)
    feat["MACD_hist"] = macd_hist

    # VIX (2)
    feat = ind.add_vix_features(feat, macro_df)

    # Calendar (2)
    dow = feat.index.dayofweek
    feat["Day_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["Day_cos"] = np.cos(2 * np.pi * dow / 5)

    # Market Regime (3)
    feat = ind.join_macro(feat, macro_df, ["SPY_return_1d", "SPY_return_5d", "SPY_vs_SMA50"])

    # Cross-Asset (3)
    feat = ind.join_macro(feat, macro_df, ["Bond_return_1d", "Dollar_return_1d", "Gold_return_1d"])

    # Momentum Dynamics (3)
    feat["RSI_3d_change"] = feat["RSI_14"] - feat["RSI_14"].shift(3)
    feat["MACD_accel"] = macd_hist - macd_hist.shift(1)
    feat["Vol_accel"] = feat["Volatility_5d"] - feat["Volatility_5d"].shift(3)

    # Lagged Signals (3)
    feat["Volume_ratio_lag1"] = feat["Volume_ratio"].shift(1)
    feat["Overnight_gap_lag1"] = feat["Overnight_gap"].shift(1)
    feat["Return_rank_5d"] = feat["Return_1d"].rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    return feat


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    """Run the daily prediction job with config/daily.toml."""
    run(PredictionJob.from_config(CONFIG, create_features, get_macro_data))


if __name__ == "__main__":
    main()
