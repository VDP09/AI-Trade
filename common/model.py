"""Training an XGBoost direction classifier and turning its output into a signal."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

TRADING_DAYS_PER_YEAR = 252


def build_dataset(df, feat, feature_cols, horizon):
    """Combine features with a binary up/down target.

    Args:
        df (pandas.DataFrame): Price data with a ``Close`` column.
        feat (pandas.DataFrame): Feature frame on the same index as ``df``.
        feature_cols (list[str]): Feature columns to keep.
        horizon (int): Trading days ahead the target looks.

    Returns:
        pandas.DataFrame: ``feature_cols`` plus ``target`` (1 if the close is
        higher ``horizon`` days later, 0 if not, NaN when not yet known) and
        ``Close``. Rows with any missing feature are dropped.
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1

    # CRITICAL: Keep target as NaN for rows where future is unknown.
    # Do NOT use .astype(int) which converts NaN → 0 (false DOWN label).
    target = pd.Series(np.nan, index=df.index)
    known = future_return.notna()
    target[known] = (future_return[known] > 0).astype(int)

    data = feat[feature_cols].copy()
    data["target"] = target
    data["Close"] = df["Close"]
    return data.dropna(subset=feature_cols)


def to_signal(pred, info):
    """Map a model prediction to a trading signal for this ticker.

    Args:
        pred (int): Model prediction, 1 = UP, 0 = DOWN.
        info (dict): Ticker info from :func:`common.tickers.classify_ticker`.

    Returns:
        tuple[str, str]: ``(signal, trade)`` where ``signal`` is ``"BUY"``,
        ``"SHORT"`` (indices only) or ``"CASH"`` (stocks only), and ``trade``
        is the human-readable action.
    """
    if pred == 1:
        return "BUY", info["up_action"]
    if info["type"] == "index":
        return "SHORT", info["down_action"]
    return "CASH", info["down_action"]


def train_and_predict(ticker, df, feat, info, feature_cols, xgb_params, horizon, train_years,
                      min_train_rows=TRADING_DAYS_PER_YEAR):
    """Train a classifier on recent history and predict the latest row.

    Args:
        ticker (str): Ticker symbol (used in the result and error messages).
        df (pandas.DataFrame): Price data with a ``Close`` column.
        feat (pandas.DataFrame): Feature frame on the same index as ``df``.
        info (dict): Ticker info from :func:`common.tickers.classify_ticker`.
        feature_cols (list[str]): Feature columns used by the model.
        xgb_params (dict): Keyword arguments for ``xgboost.XGBClassifier``.
        horizon (int): Trading days ahead being predicted.
        train_years (int): Years of most recent known-target rows to train on.
        min_train_rows (int): Minimum known-target rows required to train.

    Returns:
        dict: Prediction with keys ``ticker``, ``date`` (YYYY-MM-DD of the
        latest row), ``close``, ``signal``, ``trade``, ``up_prob``,
        ``down_prob``, ``confidence`` (max of the two), ``top_feature``,
        ``top_5`` (list of ``(feature, importance)``) and ``strategy``.

    Raises:
        ValueError: If fewer than ``min_train_rows`` trainable rows exist.
    """
    data = build_dataset(df, feat, feature_cols, horizon)

    # Exclude the last HORIZON rows from training — their targets are unknown.
    # Then take the most recent TRAIN_YEARS of *known-target* rows.
    trainable = data.iloc[:-horizon].dropna(subset=["target"])
    if len(trainable) < min_train_rows:
        raise ValueError(f"{ticker}: insufficient training data ({len(trainable)} rows, need ≥{min_train_rows})")
    train_data = trainable.tail(train_years * TRADING_DAYS_PER_YEAR)
    latest = data.iloc[[-1]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_data[feature_cols])
    X_latest_s = scaler.transform(latest[feature_cols])

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_s, train_data["target"].astype(int), verbose=False)

    pred = model.predict(X_latest_s)[0]
    down_prob, up_prob = model.predict_proba(X_latest_s)[0]
    signal, trade = to_signal(pred, info)
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)

    return {
        "ticker": ticker, "date": latest.index[0].strftime("%Y-%m-%d"),
        "close": float(latest["Close"].values[0]),
        "signal": signal, "trade": trade,
        "up_prob": float(up_prob), "down_prob": float(down_prob),
        "confidence": float(max(up_prob, down_prob)),
        "top_feature": importances[0][0],
        "top_5": importances[:5],
        "strategy": info["strategy"],
    }
