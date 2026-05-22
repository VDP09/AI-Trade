#!/usr/bin/env python3
"""
AI Daily Stock Prediction v2 — GitHub Actions Automation
Results stored in predictions.csv (committed to repo automatically).
Dashboard rendered in GitHub Actions job summary.

Edit the config section below to change tickers, data source, etc.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# ════════════════════════════════════════════════════════════════
# 👉 EDIT YOUR CONFIG HERE
# ════════════════════════════════════════════════════════════════

TICKERS_INPUT     = "^GSPC, AAPL, NVDA, MSFT"
DATA_SOURCE       = "alpaca"     # "yfinance" or "alpaca"
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")      # Set in GitHub Secrets
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")   # Set in GitHub Secrets
ALPACA_PAPER      = True           # True = paper trading, False = live
ALPACA_FEED       = "iex"          # "iex" = free tier, "sip" = paid
AUTO_TRADE        = False          # True = submit orders via Alpaca

HORIZON           = 1              # 1 trading day (tomorrow)
TRAIN_YEARS       = 3              # training window
CSV_FILE          = Path("predictions.csv")

# ════════════════════════════════════════════════════════════════
# DATA SOURCE SETUP
# ════════════════════════════════════════════════════════════════

alpaca_data_client = None
ALPACA_DATA_FEED = None

if DATA_SOURCE == "alpaca":
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_DATA_FEED = DataFeed.SIP if ALPACA_FEED == "sip" else DataFeed.IEX
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        alpaca_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        print(f"📡 Alpaca (feed={ALPACA_FEED})")
    else:
        alpaca_data_client = StockHistoricalDataClient()
        print(f"📡 Alpaca (unauthenticated)")
else:
    print("📡 Yahoo Finance")

# ════════════════════════════════════════════════════════════════
# TICKER CLASSIFICATION
# ════════════════════════════════════════════════════════════════

INDEX_MAP = {
    "^GSPC": {"long": "SPY", "short": "SH",  "name": "S&P 500",        "alpaca_proxy": "SPY"},
    "^DJI":  {"long": "DIA", "short": "DOG", "name": "Dow Jones",       "alpaca_proxy": "DIA"},
    "^IXIC": {"long": "QQQ", "short": "PSQ", "name": "Nasdaq",          "alpaca_proxy": "QQQ"},
    "^RUT":  {"long": "IWM", "short": "RWM", "name": "Russell 2000",    "alpaca_proxy": "IWM"},
    "SPY":   {"long": "SPY", "short": "SH",  "name": "S&P 500 ETF",    "alpaca_proxy": "SPY"},
    "QQQ":   {"long": "QQQ", "short": "PSQ", "name": "Nasdaq ETF",      "alpaca_proxy": "QQQ"},
    "DIA":   {"long": "DIA", "short": "DOG", "name": "Dow Jones ETF",   "alpaca_proxy": "DIA"},
    "IWM":   {"long": "IWM", "short": "RWM", "name": "Russell 2000 ETF","alpaca_proxy": "IWM"},
}

TICKERS = [t.strip().upper() for t in TICKERS_INPUT.split(",") if t.strip()]

ticker_info = {}
for t in TICKERS:
    if t in INDEX_MAP:
        ticker_info[t] = {
            "type": "index", "strategy": "long/short",
            "long_etf": INDEX_MAP[t]["long"], "short_etf": INDEX_MAP[t]["short"],
            "name": INDEX_MAP[t]["name"],
            "up_action": f"Buy {INDEX_MAP[t]['long']}",
            "down_action": f"Buy {INDEX_MAP[t]['short']}",
        }
    else:
        ticker_info[t] = {
            "type": "stock", "strategy": "long/cash",
            "long_etf": t, "short_etf": None, "name": t,
            "up_action": f"Buy {t}", "down_action": f"Sell {t} → cash",
        }

# ════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ════════════════════════════════════════════════════════════════

def download_data_yf(ticker, years=5):
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 60)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    if "Close" not in df.columns:
        for c in df.columns:
            if "close" in c.lower():
                df = df.rename(columns={c: "Close"})
                break
    return df

def download_data_alpaca(ticker, years=5):
    symbol = INDEX_MAP[ticker]["alpaca_proxy"] if ticker in INDEX_MAP else ticker
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 60)
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=start, end=end, feed=ALPACA_DATA_FEED,
    )
    df = alpaca_data_client.get_stock_bars(request).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    return df

def download_data(ticker, years=5):
    return download_data_alpaca(ticker, years) if DATA_SOURCE == "alpaca" else download_data_yf(ticker, years)

def get_macro_data(years=6):
    REQUIRED = {"VIX", "SPY", "TLT", "UUP", "GLD"}
    symbols = {"^VIX": "VIX", "SPY": "SPY", "TLT": "TLT", "UUP": "UUP", "GLD": "GLD"}
    macro = pd.DataFrame()
    for sym, label in symbols.items():
        print(f"   📥 {label}...")
        try:
            raw = download_data_yf(sym, years)
            close = raw["Close"].rename(label)
            macro = macro.join(close, how="outer") if len(macro) > 0 else pd.DataFrame(close)
        except Exception as e:
            print(f"      ⚠️ {sym}: {e}")

    # Validate required series before computing derived features
    missing = REQUIRED - set(macro.columns)
    if missing:
        raise RuntimeError(f"Missing required macro data: {sorted(missing)}. Check network/API.")
    if macro.empty:
        raise RuntimeError("Macro data frame is empty. Cannot proceed.")

    macro = macro.ffill()
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
    feat = pd.DataFrame(index=df.index)
    close, opn = df["Close"], df.get("Open", df["Close"])
    high, low = df.get("High", df["Close"]), df.get("Low", df["Close"])
    volume = df.get("Volume", pd.Series(1, index=df.index))

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
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat["ATR_14"] = tr.rolling(14).mean() / close
    feat["Intraday_range"] = (high - low) / close

    # Price Position (2)
    feat["Close_pos_5d"] = (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min() + 1e-10)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["BB_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # Volume (3)
    feat["Volume_ratio"] = volume / volume.rolling(20).mean()
    up_vol = volume * (close > close.shift(1)).astype(float)
    feat["Up_volume_ratio"] = up_vol.rolling(10).sum() / (volume.rolling(10).sum() + 1e-10)
    obv_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = pd.Series((volume * obv_sign).cumsum(), index=df.index)
    feat["OBV_slope"] = (obv - obv.shift(10)) / (volume.rolling(10).mean() * 10 + 1e-10)

    # Technical (3)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    feat["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    raw_k = 100 * (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min() + 1e-10)
    feat["Stochastic_K"] = raw_k.rolling(3).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()) / close
    feat["MACD_hist"] = macd_hist

    # VIX (2)
    feat = feat.join(macro_df[["VIX", "VIX_SMA20"]], how="left")
    feat["VIX"] = feat["VIX"].ffill()
    feat["VIX_ratio"] = feat["VIX"] / feat["VIX_SMA20"].ffill()
    feat.drop(columns=["VIX_SMA20"], inplace=True)

    # Calendar (2)
    dow = feat.index.dayofweek
    feat["Day_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["Day_cos"] = np.cos(2 * np.pi * dow / 5)

    # Market Regime (3)
    for col in ["SPY_return_1d", "SPY_return_5d", "SPY_vs_SMA50"]:
        feat = feat.join(macro_df[[col]], how="left")
        feat[col] = feat[col].ffill()

    # Cross-Asset (3)
    for col in ["Bond_return_1d", "Dollar_return_1d", "Gold_return_1d"]:
        feat = feat.join(macro_df[[col]], how="left")
        feat[col] = feat[col].ffill()

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

FEATURE_COLS = [
    "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_20d", "Overnight_gap",
    "SMA_10_50_ratio", "Price_vs_SMA200",
    "Volatility_5d", "Volatility_10d", "Volatility_20d", "Vol_ratio_5_20", "ATR_14", "Intraday_range",
    "Close_pos_5d", "BB_pct",
    "Volume_ratio", "Up_volume_ratio", "OBV_slope",
    "RSI_14", "Stochastic_K", "MACD_hist",
    "VIX", "VIX_ratio",
    "Day_sin", "Day_cos",
    "SPY_return_1d", "SPY_return_5d", "SPY_vs_SMA50",
    "Bond_return_1d", "Dollar_return_1d", "Gold_return_1d",
    "RSI_3d_change", "MACD_accel", "Vol_accel",
    "Volume_ratio_lag1", "Overnight_gap_lag1", "Return_rank_5d",
]

XGB_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=3.0, reg_lambda=4.0, min_child_weight=8, gamma=0.15,
    eval_metric="logloss", random_state=42,
)

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════

def prepare_ticker(ticker, macro_data):
    info = ticker_info[ticker]
    df = download_data(ticker, years=5)
    feat = create_features(df, macro_data)
    future_return = df["Close"].shift(-HORIZON) / df["Close"] - 1

    # CRITICAL: Keep target as NaN for rows where future is unknown.
    # Do NOT use .astype(int) which converts NaN → 0 (false DOWN label).
    target = pd.Series(np.nan, index=df.index)
    known = future_return.notna()
    target[known] = (future_return[known] > 0).astype(int)

    data = feat[FEATURE_COLS].copy()
    data["target"] = target
    data["Close"] = df["Close"]
    data = data.dropna(subset=FEATURE_COLS)

    # Exclude the last HORIZON rows from training — their targets are unknown.
    # Then take the most recent TRAIN_YEARS of *known-target* rows.
    trainable = data.iloc[:-HORIZON].dropna(subset=["target"])
    if len(trainable) < 252:
        raise ValueError(f"{ticker}: insufficient training data ({len(trainable)} rows, need ≥252)")
    train_data = trainable.tail(TRAIN_YEARS * 252)
    latest = data.iloc[[-1]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_data[FEATURE_COLS])
    X_latest_s = scaler.transform(latest[FEATURE_COLS])

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_s, train_data["target"].astype(int), verbose=False)

    pred = model.predict(X_latest_s)[0]
    proba = model.predict_proba(X_latest_s)[0]
    up_prob, down_prob = proba[1], proba[0]
    confidence = max(up_prob, down_prob)

    if pred == 1:
        signal, trade = "BUY", info["up_action"]
    elif info["type"] == "index":
        signal, trade = "SHORT", info["down_action"]
    else:
        signal, trade = "CASH", info["down_action"]

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)

    return {
        "ticker": ticker, "date": latest.index[0].strftime("%Y-%m-%d"),
        "close": float(latest["Close"].values[0]),
        "signal": signal, "trade": trade,
        "up_prob": float(up_prob), "down_prob": float(down_prob),
        "confidence": float(confidence),
        "top_feature": importances[0][0],
        "top_5": importances[:5],
        "strategy": info["strategy"],
    }

# ════════════════════════════════════════════════════════════════
# CSV STORAGE
# ════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "Date", "Ticker", "Close", "Signal", "Trade", "Strategy",
    "Confidence", "UP_Prob", "DOWN_Prob",
    "Actual_Direction", "Actual_Close", "Market_Return",
    "Strategy_Return", "Correct", "Top_Feature",
]


def load_csv():
    """Load existing CSV or create empty one.
    Uses keep_default_na=False so blank fields stay as "" not NaN.
    """
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df.fillna("")
    return pd.DataFrame(columns=CSV_COLUMNS)


def fill_past_actuals(df):
    """Fill in actual results for past predictions where HORIZON days have passed.
    Downloads each ticker's data once and reuses it across all rows.
    """
    filled = 0
    # Cache: download each ticker once, not per-row
    ticker_data_cache = {}

    for idx, row in df.iterrows():
        if row["Actual_Direction"]:  # already filled (blank = not filled)
            continue

        try:
            pred_date = pd.Timestamp(row["Date"])
        except (ValueError, TypeError):
            continue

        trading_days_since = np.busday_count(pred_date.date(), datetime.now().date())
        if trading_days_since < HORIZON:
            continue

        ticker = row["Ticker"]
        try:
            if ticker not in ticker_data_cache:
                ticker_data_cache[ticker] = download_data(ticker, years=1)
            actual_data = ticker_data_cache[ticker]
            future = actual_data[actual_data.index >= pred_date]

            if len(future) > HORIZON:
                pred_close = float(row["Close"])
                actual_close = float(future["Close"].iloc[HORIZON])
                market_return = (actual_close - pred_close) / pred_close
                # Flat days (exact same close) are extremely rare but handled explicitly
                if actual_close > pred_close:
                    actual_dir = "UP"
                elif actual_close < pred_close:
                    actual_dir = "DOWN"
                else:
                    actual_dir = "FLAT"

                signal = row["Signal"]
                if signal == "BUY":
                    strat_return = market_return
                    correct = actual_dir == "UP"
                elif signal == "SHORT":
                    strat_return = -market_return
                    correct = actual_dir == "DOWN"
                else:  # CASH
                    strat_return = 0.0
                    correct = actual_dir in ("DOWN", "FLAT")  # cash is correct if not UP

                df.at[idx, "Actual_Direction"] = actual_dir
                df.at[idx, "Actual_Close"] = f"{actual_close:.2f}"
                df.at[idx, "Market_Return"] = f"{market_return:.4f}"
                df.at[idx, "Strategy_Return"] = f"{strat_return:.4f}"
                df.at[idx, "Correct"] = "Yes" if correct else "No"
                filled += 1
        except Exception as e:
            print(f"   ⚠️ Backfill error for {ticker} on {row['Date']}: {e}")

    return df, filled


def save_predictions(df, predictions):
    """Append today's predictions to the dataframe, skip duplicates."""
    existing_keys = set()
    for _, row in df.iterrows():
        existing_keys.add(f"{row['Date']}_{row['Ticker']}")

    added = 0
    for ticker in TICKERS:
        p = predictions[ticker]
        key = f"{p['date']}_{ticker}"
        if key in existing_keys:
            print(f"   ⏩ {ticker} already logged for {p['date']}")
            continue

        new_row = {
            "Date": p["date"], "Ticker": ticker, "Close": f"{p['close']:.2f}",
            "Signal": p["signal"], "Trade": p["trade"], "Strategy": p["strategy"],
            "Confidence": f"{p['confidence']:.4f}",
            "UP_Prob": f"{p['up_prob']:.4f}", "DOWN_Prob": f"{p['down_prob']:.4f}",
            "Actual_Direction": "", "Actual_Close": "", "Market_Return": "",
            "Strategy_Return": "", "Correct": "", "Top_Feature": p["top_feature"],
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        added += 1
        print(f"   ✅ {ticker}: {p['signal']}")

    return df, added


def compute_stats(df):
    """Compute per-ticker cumulative accuracy and P&L.
    Only counts rows where Correct is explicitly 'Yes' or 'No'.
    """
    stats = {}
    for ticker in TICKERS:
        rows = df[(df["Ticker"] == ticker) & (df["Correct"].isin(["Yes", "No"]))]
        if len(rows) == 0:
            stats[ticker] = {"accuracy": None, "pnl": None, "total": 0}
            continue
        correct = (rows["Correct"] == "Yes").sum()
        total = len(rows)
        pnl = pd.to_numeric(rows["Strategy_Return"], errors="coerce").fillna(0).sum()
        stats[ticker] = {"accuracy": correct / total, "pnl": float(pnl), "total": total}
    return stats

# ════════════════════════════════════════════════════════════════
# GITHUB ACTIONS JOB SUMMARY
# ════════════════════════════════════════════════════════════════

def write_job_summary(predictions, stats):
    """Write a markdown dashboard to GitHub Actions job summary."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return  # Not running in GitHub Actions

    today = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"# 🤖 Daily Prediction — {today}",
        "",
        "## Tomorrow's Signals",
        "",
        "| Ticker | Close | Signal | Action | Strategy | Confidence |",
        "|--------|-------|--------|--------|----------|------------|",
    ]

    for ticker in TICKERS:
        p = predictions[ticker]
        sig_emoji = "🟢 BUY" if p["signal"] == "BUY" else ("🔴 SHORT" if p["signal"] == "SHORT" else "⚪ CASH")
        conf_pct = f"{p['confidence']:.1%}"
        lines.append(
            f"| **{ticker}** | ${p['close']:.2f} | {sig_emoji} | {p['trade']} | {p['strategy']} | {conf_pct} |"
        )

    # Low confidence warnings
    low_conf = [t for t in TICKERS if predictions[t]["confidence"] < 0.55]
    if low_conf:
        lines.append("")
        lines.append(f"> ⚠️ **Low confidence**: {', '.join(low_conf)} — below 55%")

    # Cumulative stats
    has_stats = any(s["total"] > 0 for s in stats.values())
    if has_stats:
        lines.extend(["", "## Cumulative Performance", "",
                       "| Ticker | Predictions | Accuracy | Cumulative P&L |",
                       "|--------|-------------|----------|----------------|"])
        for ticker in TICKERS:
            s = stats[ticker]
            if s["total"] > 0:
                acc = f"{s['accuracy']:.1%}"
                pnl = f"{s['pnl']:+.2%}"
                lines.append(f"| {ticker} | {s['total']} | {acc} | {pnl} |")
            else:
                lines.append(f"| {ticker} | 0 | — | — |")

    # Top features
    lines.extend(["", "## Top Model Importance Features", ""])
    for ticker in TICKERS:
        p = predictions[ticker]
        top3 = ", ".join(f"`{n}` ({v:.3f})" for n, v in p["top_5"][:3])
        lines.append(f"- **{ticker}**: {top3}")

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")

# ════════════════════════════════════════════════════════════════
# ALPACA TRADING
# ════════════════════════════════════════════════════════════════

def execute_trades(predictions):
    """Submit orders via Alpaca — idempotent: skips if already on the right side."""
    if DATA_SOURCE != "alpaca" or not AUTO_TRADE:
        return
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("   ❌ Missing Alpaca credentials")
        return

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = client.get_account()
    mode = "🧪 PAPER" if ALPACA_PAPER else "💰 LIVE"
    print(f"\n   {mode} | Power: ${float(account.buying_power):,.2f} | Value: ${float(account.portfolio_value):,.2f}")

    positions = {p.symbol: p for p in client.get_all_positions()}
    # Check existing open orders to avoid duplicates
    open_order_ids = set()
    try:
        for o in client.get_orders():
            if hasattr(o, "client_order_id") and o.client_order_id:
                open_order_ids.add(o.client_order_id)
    except Exception as e:
        print(f"   ❌ Could not check open orders; skipping trading for safety: {e}")
        return
    per_ticker = float(account.portfolio_value) / len(TICKERS)
    orders = 0

    for ticker in TICKERS:
        p, info = predictions[ticker], ticker_info[ticker]
        date_str = p["date"]

        # Determine desired and undesired symbols
        if p["signal"] == "BUY":
            desired_sym = info["long_etf"]
            undesired_sym = info.get("short_etf")
        elif p["signal"] == "SHORT":
            desired_sym = info["short_etf"]
            undesired_sym = info["long_etf"]
        else:  # CASH
            desired_sym = None
            undesired_sym = info["long_etf"]

        try:
            # Sell undesired side if held
            if undesired_sym and undesired_sym in positions:
                qty = abs(float(positions[undesired_sym].qty))
                sell_cid = f"pred-close-{date_str}-{ticker}-{undesired_sym}"
                if qty > 0 and sell_cid not in open_order_ids:
                    client.submit_order(MarketOrderRequest(
                        symbol=undesired_sym, qty=qty,
                        side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        client_order_id=sell_cid,
                    ))
                    print(f"   ✅ SOLD {qty:.0f} {undesired_sym}")
                    orders += 1
                elif sell_cid in open_order_ids:
                    print(f"   ⏩ {ticker}: sell order already pending ({sell_cid})")

            # Skip buy if already holding the desired side
            if desired_sym and desired_sym in positions:
                print(f"   ⏩ {ticker}: already holding {desired_sym}")
                continue

            # Buy desired side
            if desired_sym:
                buy_cid = f"pred-open-{date_str}-{ticker}-{p['signal']}"
                if buy_cid in open_order_ids:
                    print(f"   ⏩ {ticker}: order already pending ({buy_cid})")
                    continue
                client.submit_order(MarketOrderRequest(
                    symbol=desired_sym, notional=round(per_ticker, 2),
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                    client_order_id=buy_cid,
                ))
                print(f"   ✅ BUY ${per_ticker:,.0f} {desired_sym}")
                orders += 1

        except Exception as e:
            print(f"   ❌ {ticker}: {e}")

    print(f"   📊 {orders} orders submitted")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🤖 AI DAILY STOCK PREDICTION v2")
    print("=" * 60)
    print(f"   Tickers: {', '.join(TICKERS)}")
    print(f"   Features: {len(FEATURE_COLS)} | Horizon: {HORIZON}d")

    # 1. Macro data
    print(f"\n⏳ Downloading macro data...")
    macro_data = get_macro_data(years=6)
    print(f"   ✅ {macro_data.index[0].strftime('%Y-%m-%d')} → {macro_data.index[-1].strftime('%Y-%m-%d')}")

    # 2. Predictions
    print(f"\n⏳ Generating predictions...")
    predictions = {}
    for ticker in TICKERS:
        result = prepare_ticker(ticker, macro_data)
        predictions[ticker] = result
        emoji = "🟢" if result["signal"] == "BUY" else ("🔴" if result["signal"] == "SHORT" else "⚪")
        print(f"   {emoji} {ticker:<8} {result['signal']:<6} {result['confidence']:.1%}  → {result['trade']}")

    # 3. CSV: fill past actuals
    print(f"\n⏳ Updating predictions.csv...")
    df = load_csv()
    df, filled = fill_past_actuals(df)
    if filled:
        print(f"   ✅ Filled {filled} past result(s)")

    # 4. CSV: save new predictions
    df, added = save_predictions(df, predictions)
    df.to_csv(CSV_FILE, index=False)
    print(f"   📊 Added {added} prediction(s) | Total rows: {len(df)}")

    # 5. Stats
    stats = compute_stats(df)
    has_stats = any(s["total"] > 0 for s in stats.values())
    if has_stats:
        print(f"\n{'='*60}")
        print(f"  📈 CUMULATIVE PERFORMANCE (directional model P&L, not exact trade P&L)")
        print(f"{'='*60}")
        for ticker in TICKERS:
            s = stats[ticker]
            if s["total"] > 0:
                print(f"   {ticker:<8} {s['total']:>3} preds | Acc: {s['accuracy']:.1%} | P&L: {s['pnl']:+.2%}")

    # 6. Checklist
    print(f"\n{'='*60}")
    print(f"  📋 TOMORROW 9:35 AM")
    print(f"{'='*60}")
    for ticker in TICKERS:
        p = predictions[ticker]
        emoji = "🟢" if p["signal"] == "BUY" else ("🔴" if p["signal"] == "SHORT" else "⚪")
        print(f"   {emoji} {ticker:<8} → {p['trade']}")

    # 7. GitHub Actions summary
    write_job_summary(predictions, stats)

    # 8. Auto-trade
    if AUTO_TRADE:
        print(f"\n{'='*60}")
        print(f"  🤖 AUTO-TRADING")
        print(f"{'='*60}")
        try:
            execute_trades(predictions)
        except Exception as e:
            print(f"   ❌ {e}")

    print(f"\n✅ Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
