#!/usr/bin/env python3
"""
AI Weekly Stock Prediction v2 — GitHub Actions Automation
Predicts next week's direction (5 trading days) for US stocks and indices.
Results stored in weekly_predictions.csv (committed to repo automatically).

Edit the config section below to change tickers, data source, etc.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# ════════════════════════════════════════════════════════════════
# 👉 EDIT YOUR CONFIG HERE
# ════════════════════════════════════════════════════════════════

TICKERS_INPUT     = "^GSPC, AAPL, NVDA, MSFT"
DATA_SOURCE       = "alpaca"     # "yfinance" or "alpaca"
ALPACA_API_KEY    = "PK7RIZIBAN7ERZO67MMSRKVYFJ"             # Alpaca API key (leave empty for yfinance)
ALPACA_SECRET_KEY = "36LA73aV5K97xiKNcHKMUZYcKDDLLBXdxG37m4fMAx2F"             # Alpaca secret key
ALPACA_PAPER      = True           # True = paper trading, False = live
ALPACA_FEED       = "iex"          # "iex" = free tier, "sip" = paid
AUTO_TRADE        = False          # True = submit orders via Alpaca

HORIZON           = 5              # 5 trading days (1 week)
TRAIN_YEARS       = 3              # training window
CSV_FILE          = Path("weekly_predictions.csv")

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

def get_macro_data(years=7):
    """Download VIX + cross-asset data — always via yfinance."""
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
    macro = macro.ffill()

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
# 30 FEATURES — TUNED FOR WEEKLY PREDICTION
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
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df.get("High", df["Close"])
    low = df.get("Low", df["Close"])
    volume = df.get("Volume", pd.Series(1, index=df.index))

    # ── Momentum (5) — weekly-appropriate timescales ──
    feat["Return_5d"] = close.pct_change(5)
    feat["Return_10d"] = close.pct_change(10)
    feat["Return_20d"] = close.pct_change(20)
    feat["Return_60d"] = close.pct_change(60)
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
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["ATR_14"] = tr.rolling(14).mean() / close

    # ── Price Position (2) — 20-day range for weekly context ──
    roll_high_20 = high.rolling(20).max()
    roll_low_20 = low.rolling(20).min()
    feat["Close_pos_20d"] = (close - roll_low_20) / (roll_high_20 - roll_low_20 + 1e-10)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["BB_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # ── Volume (3) ──
    feat["Volume_ratio"] = volume / volume.rolling(20).mean()
    up_vol = volume * (close > close.shift(1)).astype(float)
    feat["Up_volume_ratio"] = up_vol.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
    obv_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = pd.Series((volume * obv_sign).cumsum(), index=df.index)
    feat["OBV_slope"] = (obv - obv.shift(20)) / (volume.rolling(20).mean() * 20 + 1e-10)

    # ── Technical (3) — RSI and MACD are effective at weekly ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    feat["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line) / close
    feat["MACD_hist"] = macd_hist
    feat["RSI_5d_change"] = feat["RSI_14"] - feat["RSI_14"].shift(5)

    # ── VIX (3) — fear gauge with weekly change ──
    feat = feat.join(macro_df[["VIX", "VIX_SMA20", "VIX_weekly_chg"]], how="left")
    feat["VIX"] = feat["VIX"].ffill()
    feat["VIX_ratio"] = feat["VIX"] / feat["VIX_SMA20"].ffill()
    feat["VIX_weekly_chg"] = feat["VIX_weekly_chg"].ffill()
    feat.drop(columns=["VIX_SMA20"], inplace=True)

    # ── Market Regime (3) — broad market weekly context ──
    for col in ["SPY_return_5d", "SPY_return_20d", "SPY_vs_SMA50"]:
        feat = feat.join(macro_df[[col]], how="left")
        feat[col] = feat[col].ffill()

    # ── Cross-Asset (3) — weekly returns match HORIZON ──
    for col in ["Bond_return_5d", "Dollar_return_5d", "Gold_return_5d"]:
        feat = feat.join(macro_df[[col]], how="left")
        feat[col] = feat[col].ffill()

    # ── Momentum Dynamics (2) — weekly-scale rate of change ──
    feat["MACD_accel_5d"] = macd_hist - macd_hist.shift(5)
    feat["Vol_accel"] = feat["Volatility_20d"] - feat["Volatility_20d"].shift(5)

    return feat

FEATURE_COLS = [
    # Momentum (5)
    "Return_5d", "Return_10d", "Return_20d", "Return_60d", "Return_5d_lag1",
    # Trend (3)
    "SMA_10_50_ratio", "SMA_50_200_ratio", "Price_vs_SMA200",
    # Volatility (4)
    "Volatility_20d", "Volatility_60d", "Vol_ratio_20_60", "ATR_14",
    # Price Position (2)
    "Close_pos_20d", "BB_pct",
    # Volume (3)
    "Volume_ratio", "Up_volume_ratio", "OBV_slope",
    # Technical (3)
    "RSI_14", "MACD_hist", "RSI_5d_change",
    # VIX (3)
    "VIX", "VIX_ratio", "VIX_weekly_chg",
    # Market Regime (3)
    "SPY_return_5d", "SPY_return_20d", "SPY_vs_SMA50",
    # Cross-Asset (3)
    "Bond_return_5d", "Dollar_return_5d", "Gold_return_5d",
    # Momentum Dynamics (2)
    "MACD_accel_5d", "Vol_accel",
]

# XGBoost — weekly is less noisy than daily, needs less regularization
XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.5,
    reg_lambda=2.5,
    min_child_weight=5,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════

def prepare_ticker(ticker, macro_data):
    info = ticker_info[ticker]
    df = download_data(ticker, years=5)
    feat = create_features(df, macro_data)
    future_return = df["Close"].shift(-HORIZON) / df["Close"] - 1
    target = (future_return > 0).astype(int)

    data = feat[FEATURE_COLS].copy()
    data["target"] = target
    data["Close"] = df["Close"]
    data = data.dropna(subset=FEATURE_COLS)

    train_data = data[data["target"].notna()].tail(TRAIN_YEARS * 252)
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
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE, dtype=str)
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=CSV_COLUMNS)

def fill_past_actuals(df):
    filled = 0
    for idx, row in df.iterrows():
        if row["Actual_Direction"]:
            continue
        try:
            pred_date = pd.Timestamp(row["Date"])
        except:
            continue
        trading_days_since = np.busday_count(pred_date.date(), datetime.now().date())
        if trading_days_since < HORIZON:
            continue
        try:
            ticker = row["Ticker"]
            actual_data = download_data(ticker, years=1)
            future = actual_data[actual_data.index >= pred_date]
            if len(future) > HORIZON:
                pred_close = float(row["Close"])
                actual_close = float(future["Close"].iloc[HORIZON])
                market_return = (actual_close - pred_close) / pred_close
                actual_dir = "UP" if actual_close > pred_close else "DOWN"
                signal = row["Signal"]
                if signal == "BUY":
                    strat_return = market_return
                    correct = actual_dir == "UP"
                elif signal == "SHORT":
                    strat_return = -market_return
                    correct = actual_dir == "DOWN"
                else:
                    strat_return = 0.0
                    correct = actual_dir == "DOWN"
                df.at[idx, "Actual_Direction"] = actual_dir
                df.at[idx, "Actual_Close"] = f"{actual_close:.2f}"
                df.at[idx, "Market_Return"] = f"{market_return:.4f}"
                df.at[idx, "Strategy_Return"] = f"{strat_return:.4f}"
                df.at[idx, "Correct"] = "Yes" if correct else "No"
                filled += 1
        except Exception:
            pass
    return df, filled

def save_predictions(df, predictions):
    existing_keys = {f"{r['Date']}_{r['Ticker']}" for _, r in df.iterrows()}
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
    stats = {}
    for ticker in TICKERS:
        rows = df[(df["Ticker"] == ticker) & (df["Correct"] != "")]
        if len(rows) == 0:
            stats[ticker] = {"accuracy": None, "pnl": None, "total": 0}
            continue
        correct = (rows["Correct"] == "Yes").sum()
        total = len(rows)
        pnl = rows["Strategy_Return"].apply(lambda x: float(x) if x else 0).sum()
        stats[ticker] = {"accuracy": correct / total, "pnl": pnl, "total": total}
    return stats

# ════════════════════════════════════════════════════════════════
# GITHUB ACTIONS JOB SUMMARY
# ════════════════════════════════════════════════════════════════

def write_job_summary(predictions, stats):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    today = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"# 🤖 Weekly Prediction — {today}",
        "",
        "## Next Week's Signals",
        "",
        "| Ticker | Close | Signal | Monday Action | Strategy | Confidence |",
        "|--------|-------|--------|---------------|----------|------------|",
    ]
    for ticker in TICKERS:
        p = predictions[ticker]
        sig = "🟢 BUY" if p["signal"] == "BUY" else ("🔴 SHORT" if p["signal"] == "SHORT" else "⚪ CASH")
        lines.append(f"| **{ticker}** | ${p['close']:.2f} | {sig} | {p['trade']} | {p['strategy']} | {p['confidence']:.1%} |")

    low_conf = [t for t in TICKERS if predictions[t]["confidence"] < 0.55]
    if low_conf:
        lines += ["", f"> ⚠️ **Low confidence**: {', '.join(low_conf)} — below 55%"]

    has_stats = any(s["total"] > 0 for s in stats.values())
    if has_stats:
        lines += ["", "## Cumulative Performance", "",
                   "| Ticker | Weeks | Accuracy | Cumulative P&L |",
                   "|--------|-------|----------|----------------|"]
        for ticker in TICKERS:
            s = stats[ticker]
            if s["total"] > 0:
                lines.append(f"| {ticker} | {s['total']} | {s['accuracy']:.1%} | {s['pnl']:+.2%} |")
            else:
                lines.append(f"| {ticker} | 0 | — | — |")

    lines += ["", "## Top Driving Features", ""]
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
    per_ticker = float(account.portfolio_value) / len(TICKERS)
    orders = 0

    for ticker in TICKERS:
        p, info = predictions[ticker], ticker_info[ticker]
        try:
            sell_sym = info.get("short_etf") if p["signal"] == "BUY" else info["long_etf"]
            buy_sym = info["long_etf"] if p["signal"] == "BUY" else (info["short_etf"] if p["signal"] == "SHORT" else None)

            if sell_sym and sell_sym in positions:
                qty = abs(float(positions[sell_sym].qty))
                if qty > 0:
                    client.submit_order(MarketOrderRequest(symbol=sell_sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                    print(f"   ✅ SOLD {qty:.0f} {sell_sym}")
                    orders += 1
            if buy_sym:
                client.submit_order(MarketOrderRequest(symbol=buy_sym, notional=round(per_ticker, 2), side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                print(f"   ✅ BUY ${per_ticker:,.0f} {buy_sym}")
                orders += 1
        except Exception as e:
            print(f"   ❌ {ticker}: {e}")

    print(f"   📊 {orders} orders submitted")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🤖 AI WEEKLY STOCK PREDICTION v2 (30 features)")
    print("=" * 60)
    print(f"   Tickers: {', '.join(TICKERS)}")
    print(f"   Features: {len(FEATURE_COLS)} | Horizon: {HORIZON}d (1 week)")

    # 1. Macro data
    print(f"\n⏳ Downloading macro data...")
    macro_data = get_macro_data(years=7)
    print(f"   ✅ {macro_data.index[0].strftime('%Y-%m-%d')} → {macro_data.index[-1].strftime('%Y-%m-%d')}")

    # 2. Predictions
    print(f"\n⏳ Generating weekly predictions...")
    predictions = {}
    for ticker in TICKERS:
        result = prepare_ticker(ticker, macro_data)
        predictions[ticker] = result
        emoji = "🟢" if result["signal"] == "BUY" else ("🔴" if result["signal"] == "SHORT" else "⚪")
        print(f"   {emoji} {ticker:<8} {result['signal']:<6} {result['confidence']:.1%}  → {result['trade']}")

    # 3. CSV: fill past actuals
    print(f"\n⏳ Updating weekly_predictions.csv...")
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
        print(f"  📈 CUMULATIVE PERFORMANCE")
        print(f"{'='*60}")
        for ticker in TICKERS:
            s = stats[ticker]
            if s["total"] > 0:
                print(f"   {ticker:<8} {s['total']:>3} weeks | Acc: {s['accuracy']:.1%} | P&L: {s['pnl']:+.2%}")

    # 6. Checklist
    print(f"\n{'='*60}")
    print(f"  📋 MONDAY 9:35 AM TRADE CHECKLIST")
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
