"""Prediction CSV log: load, back-fill actual outcomes, append, and summarize.

All values in the log are kept as strings; blank fields mean "not yet known".
"""

from datetime import datetime

import numpy as np
import pandas as pd

CSV_COLUMNS = [
    "Date", "Ticker", "Close", "Signal", "Trade", "Strategy",
    "Confidence", "UP_Prob", "DOWN_Prob",
    "Actual_Direction", "Actual_Close", "Market_Return",
    "Strategy_Return", "Correct", "Top_Feature",
]


def load_csv(csv_file):
    """Load the prediction log, or start an empty one.

    Uses ``keep_default_na=False`` so blank fields stay as ``""`` not NaN.

    Args:
        csv_file (pathlib.Path): Path to the CSV log.

    Returns:
        pandas.DataFrame: All-string log with every column in
        ``CSV_COLUMNS`` (missing columns are added as blanks).
    """
    if csv_file.exists():
        df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df.fillna("")
    return pd.DataFrame(columns=CSV_COLUMNS)


def evaluate_outcome(signal, pred_close, actual_close):
    """Score a past prediction against what actually happened.

    Args:
        signal (str): ``"BUY"``, ``"SHORT"`` or ``"CASH"``.
        pred_close (float): Close on the prediction date.
        actual_close (float): Close ``horizon`` trading days later.

    Returns:
        tuple[str, float, float, bool]: ``(actual_direction, market_return,
        strategy_return, correct)`` where ``actual_direction`` is ``"UP"``,
        ``"DOWN"`` or ``"FLAT"``. CASH earns 0 and is correct unless the
        market went UP.
    """
    market_return = (actual_close - pred_close) / pred_close
    # Flat days (exact same close) are extremely rare but handled explicitly
    if actual_close > pred_close:
        actual_dir = "UP"
    elif actual_close < pred_close:
        actual_dir = "DOWN"
    else:
        actual_dir = "FLAT"

    if signal == "BUY":
        return actual_dir, market_return, market_return, actual_dir == "UP"
    if signal == "SHORT":
        return actual_dir, market_return, -market_return, actual_dir == "DOWN"
    # CASH is correct if not UP
    return actual_dir, market_return, 0.0, actual_dir in ("DOWN", "FLAT")


def fill_past_actuals(df, horizon, download_data, years=1):
    """Fill in actual results for past predictions whose horizon has passed.

    Downloads each ticker's data once and reuses it across all rows. Rows
    that are already filled, have an invalid date, or are too recent are
    skipped; download errors are logged and skipped.

    Args:
        df (pandas.DataFrame): Prediction log from :func:`load_csv`.
            Modified in place.
        horizon (int): Trading days between prediction and outcome.
        download_data (Callable[[str, int], pandas.DataFrame]): Price
            downloader, called as ``download_data(ticker, years=years)``.
        years (int): Years of price history to download per ticker.

    Returns:
        tuple[pandas.DataFrame, int]: The updated log and the number of rows
        filled.
    """
    filled = 0
    ticker_data_cache = {}

    for idx, row in df.iterrows():
        if row["Actual_Direction"]:  # already filled (blank = not filled)
            continue
        try:
            pred_date = pd.Timestamp(row["Date"])
        except (ValueError, TypeError):
            continue
        if np.busday_count(pred_date.date(), datetime.now().date()) < horizon:
            continue

        ticker = row["Ticker"]
        try:
            if ticker not in ticker_data_cache:
                ticker_data_cache[ticker] = download_data(ticker, years=years)
            actual_data = ticker_data_cache[ticker]
            future = actual_data[actual_data.index >= pred_date]
            if len(future) <= horizon:
                continue

            actual_close = float(future["Close"].iloc[horizon])
            actual_dir, market_return, strat_return, correct = evaluate_outcome(
                row["Signal"], float(row["Close"]), actual_close)

            df.at[idx, "Actual_Direction"] = actual_dir
            df.at[idx, "Actual_Close"] = f"{actual_close:.2f}"
            df.at[idx, "Market_Return"] = f"{market_return:.4f}"
            df.at[idx, "Strategy_Return"] = f"{strat_return:.4f}"
            df.at[idx, "Correct"] = "Yes" if correct else "No"
            filled += 1
        except Exception as e:
            print(f"   ⚠️ Backfill error for {ticker} on {row['Date']}: {e}")

    return df, filled


def prediction_row(p):
    """Format a prediction as a CSV log row.

    Args:
        p (dict): Prediction from :func:`common.model.train_and_predict`.

    Returns:
        dict[str, str]: Row keyed by ``CSV_COLUMNS`` with formatted values and
        blank outcome fields.
    """
    return {
        "Date": p["date"], "Ticker": p["ticker"], "Close": f"{p['close']:.2f}",
        "Signal": p["signal"], "Trade": p["trade"], "Strategy": p["strategy"],
        "Confidence": f"{p['confidence']:.4f}",
        "UP_Prob": f"{p['up_prob']:.4f}", "DOWN_Prob": f"{p['down_prob']:.4f}",
        "Actual_Direction": "", "Actual_Close": "", "Market_Return": "",
        "Strategy_Return": "", "Correct": "", "Top_Feature": p["top_feature"],
    }


def save_predictions(df, predictions, tickers):
    """Append new predictions to the log, skipping duplicates.

    A prediction is a duplicate if the log already has a row with the same
    Date and Ticker.

    Args:
        df (pandas.DataFrame): Prediction log.
        predictions (dict[str, dict]): Predictions keyed by ticker.
        tickers (list[str]): Tickers to append, in order.

    Returns:
        tuple[pandas.DataFrame, int]: The updated log and the number of rows
        added.
    """
    existing_keys = {f"{r['Date']}_{r['Ticker']}" for _, r in df.iterrows()}
    added = 0
    for ticker in tickers:
        p = predictions[ticker]
        if f"{p['date']}_{ticker}" in existing_keys:
            print(f"   ⏩ {ticker} already logged for {p['date']}")
            continue
        df = pd.concat([df, pd.DataFrame([prediction_row(p)])], ignore_index=True)
        added += 1
        print(f"   ✅ {ticker}: {p['signal']}")
    return df, added


def compute_stats(df, tickers):
    """Compute per-ticker cumulative accuracy and P&L.

    Only counts rows where Correct is explicitly ``"Yes"`` or ``"No"``.

    Args:
        df (pandas.DataFrame): Prediction log.
        tickers (list[str]): Tickers to summarize.

    Returns:
        dict[str, dict]: Per ticker, ``{"accuracy", "pnl", "total"}`` where
        ``pnl`` is the sum of strategy returns. ``accuracy`` and ``pnl`` are
        None when ``total`` is 0.
    """
    stats = {}
    for ticker in tickers:
        rows = df[(df["Ticker"] == ticker) & (df["Correct"].isin(["Yes", "No"]))]
        if len(rows) == 0:
            stats[ticker] = {"accuracy": None, "pnl": None, "total": 0}
            continue
        correct = (rows["Correct"] == "Yes").sum()
        total = len(rows)
        pnl = pd.to_numeric(rows["Strategy_Return"], errors="coerce").fillna(0).sum()
        stats[ticker] = {"accuracy": correct / total, "pnl": float(pnl), "total": total}
    return stats
