"""Technical indicator helpers shared by the daily and weekly feature sets.

Every indicator takes and returns ``pandas.Series`` aligned on the price
index; the first ``window`` values are NaN while the rolling window fills.
"""

import numpy as np
import pandas as pd

EPS = 1e-10  # Guards against division by zero in range/volume ratios


def ohlcv(df):
    """Extract price and volume series, with fallbacks for missing columns.

    Args:
        df (pandas.DataFrame): Price data with at least a ``Close`` column.

    Returns:
        tuple[pandas.Series, ...]: ``(open, high, low, close, volume)``.
        Missing ``Open``/``High``/``Low`` fall back to ``Close``; missing
        ``Volume`` falls back to a constant 1.
    """
    close = df["Close"]
    return (
        df.get("Open", close),
        df.get("High", close),
        df.get("Low", close),
        close,
        df.get("Volume", pd.Series(1, index=df.index)),
    )


def atr_pct(high, low, close, window=14):
    """Average True Range as a fraction of price.

    Args:
        high (pandas.Series): Daily highs.
        low (pandas.Series): Daily lows.
        close (pandas.Series): Daily closes.
        window (int): Averaging window in days.

    Returns:
        pandas.Series: Rolling mean of the true range divided by close.
    """
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean() / close


def close_position(close, high, low, window):
    """Locate the close within the rolling high/low range.

    Args:
        close (pandas.Series): Daily closes.
        high (pandas.Series): Daily highs.
        low (pandas.Series): Daily lows.
        window (int): Lookback window in days.

    Returns:
        pandas.Series: 0 at the rolling low, 1 at the rolling high.
    """
    lo, hi = low.rolling(window).min(), high.rolling(window).max()
    return (close - lo) / (hi - lo + EPS)


def bollinger_pct(close, window=20):
    """Bollinger %B: position of the close within the 2-sigma bands.

    Args:
        close (pandas.Series): Daily closes.
        window (int): Moving-average window in days.

    Returns:
        pandas.Series: 0 at the lower band, 1 at the upper band (can go
        outside [0, 1] when price breaks a band).
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - (mid - 2 * std)) / (4 * std + EPS)


def up_volume_ratio(close, volume, window):
    """Share of volume traded on up days.

    Args:
        close (pandas.Series): Daily closes.
        volume (pandas.Series): Daily volume.
        window (int): Lookback window in days.

    Returns:
        pandas.Series: Up-day volume divided by total volume over the window,
        in [0, 1].
    """
    up_vol = volume * (close > close.shift(1)).astype(float)
    return up_vol.rolling(window).sum() / (volume.rolling(window).sum() + EPS)


def obv_slope(close, volume, window):
    """Change in On-Balance Volume, normalized by average volume.

    Args:
        close (pandas.Series): Daily closes.
        volume (pandas.Series): Daily volume.
        window (int): Lookback window in days.

    Returns:
        pandas.Series: OBV change over ``window`` days divided by the total
        average volume over the same window (roughly in [-1, 1]).
    """
    prev = close.shift(1)
    sign = np.where(close > prev, 1, np.where(close < prev, -1, 0))
    obv = pd.Series((volume * sign).cumsum(), index=close.index)
    return (obv - obv.shift(window)) / (volume.rolling(window).mean() * window + EPS)


def rsi(close, window=14):
    """Relative Strength Index (simple moving-average variant).

    Args:
        close (pandas.Series): Daily closes.
        window (int): Averaging window in days.

    Returns:
        pandas.Series: RSI in [0, 100]; NaN where there were no losses in
        the window.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def macd_hist(close):
    """MACD(12, 26, 9) histogram as a fraction of price.

    Args:
        close (pandas.Series): Daily closes.

    Returns:
        pandas.Series: (MACD line - signal line) / close.
    """
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return (line - line.ewm(span=9, adjust=False).mean()) / close


def join_macro(feat, macro_df, cols):
    """Left-join macro columns onto the feature frame and forward-fill them.

    Args:
        feat (pandas.DataFrame): Feature frame indexed by date.
        macro_df (pandas.DataFrame): Macro data indexed by date.
        cols (list[str]): Macro columns to add.

    Returns:
        pandas.DataFrame: ``feat`` with ``cols`` added, aligned to its index
        and forward-filled over gaps (e.g. differing holidays).
    """
    feat = feat.join(macro_df[cols], how="left")
    feat[cols] = feat[cols].ffill()
    return feat


def add_vix_features(feat, macro_df, extra_cols=()):
    """Add VIX level, VIX ratio to its 20-day SMA, and optional extra columns.

    Args:
        feat (pandas.DataFrame): Feature frame indexed by date.
        macro_df (pandas.DataFrame): Macro data containing ``VIX``,
            ``VIX_SMA20`` and any ``extra_cols``.
        extra_cols (Iterable[str]): Additional macro columns to join, e.g.
            ``["VIX_weekly_chg"]``.

    Returns:
        pandas.DataFrame: ``feat`` with ``VIX``, ``VIX_ratio`` and
        ``extra_cols`` added (``VIX_SMA20`` is used then dropped).
    """
    feat = join_macro(feat, macro_df, ["VIX", "VIX_SMA20", *extra_cols])
    feat["VIX_ratio"] = feat["VIX"] / feat["VIX_SMA20"]
    return feat.drop(columns=["VIX_SMA20"])
