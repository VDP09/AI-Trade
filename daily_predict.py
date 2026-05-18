#!/usr/bin/env python3
"""
AI Daily Stock Prediction System v2 — Standalone Script
Runs via GitHub Actions (or cron/manually). No Colab dependency.

Config via environment variables:
  TICKERS                       comma-separated (default: "^GSPC,AAPL,NVDA,MSFT")
  DATA_SOURCE                   "yfinance" or "alpaca" (default: yfinance)
  ALPACA_API_KEY                Alpaca API key
  ALPACA_SECRET_KEY             Alpaca secret
  ALPACA_PAPER                  "true"/"false" (default: true)
  ALPACA_FEED                   "iex"/"sip" (default: iex)
  AUTO_TRADE                    "true"/"false" (default: false)
  GOOGLE_SERVICE_ACCOUNT_JSON   Service account JSON (for Sheets logging)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

import yfinance as yf  # Always needed for VIX + cross-asset

# ════════════════════════════════════════════════════════════════
# CONFIGURATION — from environment variables
# ════════════════════════════════════════════════════════════════

TICKERS_INPUT = os.environ.get("TICKERS", "^GSPC,AAPL,NVDA,MSFT")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance").lower()
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
ALPACA_FEED = os.environ.get("ALPACA_FEED", "iex").lower()
AUTO_TRADE = os.environ.get("AUTO_TRADE", "false").lower() == "true"
GOOGLE_SA_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")

HORIZON = 1
TRAIN_YEARS = 3
SHEET_NAME = "AI Daily Predictions v2"

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
        print(f"📡 Data source: Alpaca (feed={ALPACA_FEED})")
    else:
        alpaca_data_client = StockHistoricalDataClient()
        print(f"📡 Data source: Alpaca (unauthenticated, feed={ALPACA_FEED})")
else:
    print("📡 Data source: Yahoo Finance")

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

print(f"\n📊 Tickers: {', '.join(TICKERS)} (HORIZON={HORIZON})")

# ════════════════════════════════════════════════════════════════
# DATA DOWNLOAD FUNCTIONS
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
    symbol = ticker
    if ticker in INDEX_MAP:
        symbol = INDEX_MAP[ticker]["alpaca_proxy"]
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 60)
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
        start=start, end=end, feed=ALPACA_DATA_FEED,
    )
    bars = alpaca_data_client.get_stock_bars(request)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    return df


def download_data(ticker, years=5):
    if DATA_SOURCE == "yfinance":
        return download_data_yf(ticker, years)
    else:
        return download_data_alpaca(ticker, years)


def get_macro_data(years=6):
    """Download VIX + cross-asset data — always via yfinance."""
    symbols = {"^VIX": "VIX", "SPY": "SPY", "TLT": "TLT", "UUP": "UUP", "GLD": "GLD"}
    macro = pd.DataFrame()
    for sym, label in symbols.items():
        print(f"   📥 {label} ({sym})...")
        try:
            raw = download_data_yf(sym, years)
            close = raw["Close"].rename(label)
            macro = macro.join(close, how="outer") if len(macro) > 0 else pd.DataFrame(close)
        except Exception as e:
            print(f"      ⚠️ Failed: {e}")
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
# FEATURE ENGINEERING — 37 FEATURES
# ════════════════════════════════════════════════════════════════

def create_features(df, macro_df):
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    opn = df["Open"] if "Open" in df.columns else close
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

    # Momentum (6)
    feat["Return_1d"] = close.pct_change(1)
    feat["Return_2d"] = close.pct_change(2)
    feat["Return_3d"] = close.pct_change(3)
    feat["Return_5d"] = close.pct_change(5)
    feat["Return_20d"] = close.pct_change(20)
    feat["Overnight_gap"] = opn / close.shift(1) - 1

    # Trend (2)
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["SMA_10_50_ratio"] = sma10 / sma50
    feat["Price_vs_SMA200"] = close / sma200

    # Volatility (6)
    daily_ret = close.pct_change()
    feat["Volatility_5d"] = daily_ret.rolling(5).std()
    feat["Volatility_10d"] = daily_ret.rolling(10).std()
    feat["Volatility_20d"] = daily_ret.rolling(20).std()
    feat["Vol_ratio_5_20"] = feat["Volatility_5d"] / feat["Volatility_20d"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["ATR_14"] = tr.rolling(14).mean() / close
    feat["Intraday_range"] = (high - low) / close

    # Price Position (2)
    roll_high_5 = high.rolling(5).max()
    roll_low_5 = low.rolling(5).min()
    feat["Close_pos_5d"] = (close - roll_low_5) / (roll_high_5 - roll_low_5 + 1e-10)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["BB_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # Volume (3)
    feat["Volume_ratio"] = volume / volume.rolling(20).mean()
    up_vol = volume * (close > close.shift(1)).astype(float)
    feat["Up_volume_ratio"] = up_vol.rolling(10).sum() / (volume.rolling(10).sum() + 1e-10)
    obv_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = (volume * obv_sign).cumsum()
    obv_series = pd.Series(obv, index=df.index)
    feat["OBV_slope"] = (obv_series - obv_series.shift(10)) / (volume.rolling(10).mean() * 10 + 1e-10)

    # Technical (3)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["RSI_14"] = 100 - (100 / (1 + rs))

    low_5 = low.rolling(5).min()
    high_5 = high.rolling(5).max()
    raw_k = 100 * (close - low_5) / (high_5 - low_5 + 1e-10)
    feat["Stochastic_K"] = raw_k.rolling(3).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line) / close
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
    ret_1d = feat["Return_1d"]
    feat["Return_rank_5d"] = ret_1d.rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

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

# XGBoost — tuned for daily + 37 features
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=3.0,
    reg_lambda=4.0,
    min_child_weight=8,
    gamma=0.15,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════

def prepare_ticker(ticker, macro_data):
    """Train model on most recent 3 years, predict tomorrow."""
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

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    current_values = {f: float(latest[f].values[0]) for f in FEATURE_COLS}

    return {
        "ticker": ticker,
        "date": latest.index[0].strftime("%Y-%m-%d"),
        "close": float(latest["Close"].values[0]),
        "prediction": int(pred),
        "signal": signal,
        "trade": trade,
        "up_prob": float(up_prob),
        "down_prob": float(down_prob),
        "confidence": float(confidence),
        "top_feature": sorted_features[0][0],
        "top_5_features": sorted_features[:5],
        "current_values": current_values,
        "strategy": info["strategy"],
    }

# ════════════════════════════════════════════════════════════════
# GOOGLE SHEETS LOGGING (service account auth)
# ════════════════════════════════════════════════════════════════

def log_to_sheets(predictions):
    """Log predictions to Google Sheets using service account."""
    if not GOOGLE_SA_JSON:
        print("\n⚠️  GOOGLE_SERVICE_ACCOUNT_JSON not set — skipping Sheets logging.")
        return

    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    sa_info = json.loads(GOOGLE_SA_JSON)
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    gc = gspread.authorize(creds)

    HEADERS = [
        "Date", "Ticker", "Close", "Signal", "Trade", "Strategy",
        "Confidence", "UP Prob", "DOWN Prob", "Actual Direction",
        "Actual Close", "Actual Return", "Strategy Return", "Correct?",
        "Cumulative Accuracy", "Cumulative P&L", "Top Feature",
    ]

    # Open or create sheet
    try:
        sh = gc.open(SHEET_NAME)
        ws = sh.sheet1
        print(f"   📄 Opened: {SHEET_NAME}")
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
        ws = sh.sheet1
        ws.update("A1", [HEADERS])
        ws.format("A1:Q1", {
            "backgroundColor": {"red": 0.1, "green": 0.1, "blue": 0.2},
            "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1},
                           "bold": True, "fontSize": 11},
            "horizontalAlignment": "CENTER",
        })
        col_widths = [100, 70, 80, 70, 160, 100, 90, 80, 80, 100, 90, 100, 100, 70, 120, 100, 140]
        requests = []
        for idx, w in enumerate(col_widths):
            requests.append({
                "updateDimensionProperties": {
                    "range": {"sheetId": ws.id, "dimension": "COLUMNS",
                              "startIndex": idx, "endIndex": idx + 1},
                    "properties": {"pixelSize": w}, "fields": "pixelSize",
                }
            })
        sh.batch_update({"requests": requests})
        # Share with your personal Google account if needed
        # sh.share("your-email@gmail.com", perm_type="user", role="writer")
        print(f"   📄 Created: {SHEET_NAME}")

    # ── Fill past actuals ──
    all_rows = ws.get_all_values()
    if len(all_rows) > 1:
        print("   ⏳ Checking past predictions for actuals...")
        cum_correct, cum_total, cum_pnl = {}, {}, {}
        filled_count = 0

        for row_idx in range(1, len(all_rows)):
            row = all_rows[row_idx]
            if len(row) < 17:
                row.extend([""] * (17 - len(row)))

            date_str, row_ticker, row_signal = row[0], row[1], row[3]
            actual_dir = row[9]

            if actual_dir:
                if row_ticker not in cum_total:
                    cum_total[row_ticker] = 0; cum_correct[row_ticker] = 0; cum_pnl[row_ticker] = 0
                cum_total[row_ticker] += 1
                if row[13] in ["Yes", "TRUE", "1"]:
                    cum_correct[row_ticker] += 1
                try:
                    cum_pnl[row_ticker] += float(row[12].replace("%", "")) / 100
                except:
                    pass
                continue

            try:
                pred_date = pd.Timestamp(date_str)
            except:
                continue

            trading_days_since = np.busday_count(pred_date.date(), datetime.now().date())
            if trading_days_since < HORIZON:
                continue

            try:
                actual_data = download_data(row_ticker, years=1)
                mask = actual_data.index >= pred_date
                future_data = actual_data[mask]
                if len(future_data) > HORIZON:
                    pred_close = float(row[2])
                    actual_close = float(future_data["Close"].iloc[HORIZON])
                    market_return = (actual_close - pred_close) / pred_close
                    actual_direction = "UP" if actual_close > pred_close else "DOWN"

                    if row_signal == "BUY":
                        strategy_return = market_return
                        correct = actual_direction == "UP"
                    elif row_signal == "SHORT":
                        strategy_return = -market_return
                        correct = actual_direction == "DOWN"
                    else:
                        strategy_return = 0
                        correct = actual_direction == "DOWN"

                    if row_ticker not in cum_total:
                        cum_total[row_ticker] = 0
                        cum_correct[row_ticker] = 0
                        cum_pnl[row_ticker] = 0
                    cum_total[row_ticker] += 1
                    if correct:
                        cum_correct[row_ticker] += 1
                    cum_pnl[row_ticker] += strategy_return
                    cum_acc = cum_correct[row_ticker] / cum_total[row_ticker]

                    cell_row = row_idx + 1
                    ws.update(f"J{cell_row}:Q{cell_row}", [[
                        actual_direction, f"{actual_close:.2f}", f"{market_return:.2%}",
                        f"{strategy_return:.2%}", "Yes" if correct else "No",
                        f"{cum_acc:.1%}", f"{cum_pnl[row_ticker]:.2%}",
                        row[16] if len(row) > 16 else "",
                    ]])

                    bg = ({"red": 0.85, "green": 0.95, "blue": 0.85} if correct else
                          {"red": 1, "green": 1, "blue": 0.88} if row_signal == "CASH" else
                          {"red": 0.95, "green": 0.85, "blue": 0.85})
                    ws.format(f"A{cell_row}:Q{cell_row}", {"backgroundColor": bg})
                    filled_count += 1
            except Exception:
                pass

        if filled_count:
            print(f"   ✅ Filled {filled_count} past prediction(s)")
        else:
            print("   ℹ️  No past predictions ready yet")

    # ── Log today's predictions ──
    print("   ⏳ Logging today's predictions...")
    all_rows = ws.get_all_values()
    existing_keys = {f"{r[0]}_{r[1]}" for r in all_rows[1:] if len(r) >= 2}

    logged = 0
    for ticker in TICKERS:
        p = predictions[ticker]
        key = f"{p['date']}_{ticker}"
        if key in existing_keys:
            print(f"      ⏩ {ticker} already logged for {p['date']}")
            continue

        new_row = [
            p["date"], ticker, f"{p['close']:.2f}", p["signal"], p["trade"],
            p["strategy"], f"{p['confidence']:.1%}", f"{p['up_prob']:.1%}",
            f"{p['down_prob']:.1%}", "", "", "", "", "", "", "", p["top_feature"],
        ]
        ws.append_row(new_row, value_input_option="USER_ENTERED")
        cell_row = len(ws.get_all_values())

        sig_bg = ({"red": 0.83, "green": 0.93, "blue": 0.83} if p["signal"] == "BUY" else
                  {"red": 0.97, "green": 0.84, "blue": 0.86} if p["signal"] == "SHORT" else
                  {"red": 0.89, "green": 0.89, "blue": 0.9})
        ws.format(f"D{cell_row}", {"backgroundColor": sig_bg, "textFormat": {"bold": True}})
        logged += 1
        print(f"      ✅ {ticker}: {p['signal']}")

    print(f"   📊 Logged {logged} new prediction(s)")
    print(f"   🔗 {sh.url}")

# ════════════════════════════════════════════════════════════════
# ALPACA TRADING
# ════════════════════════════════════════════════════════════════

def execute_trades(predictions):
    """Submit orders via Alpaca API."""
    if DATA_SOURCE != "alpaca" or not AUTO_TRADE:
        return
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("\n❌ Cannot trade — missing Alpaca credentials.")
        return

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = trading_client.get_account()
    mode = "🧪 PAPER" if ALPACA_PAPER else "💰 LIVE"

    print(f"\n{'='*60}")
    print(f"  {mode} AUTO-TRADING")
    print(f"{'='*60}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Portfolio: ${float(account.portfolio_value):,.2f}")

    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    per_ticker = float(account.portfolio_value) / len(TICKERS)
    order_count = 0

    for ticker in TICKERS:
        p = predictions[ticker]
        info = ticker_info[ticker]

        try:
            if p["signal"] == "BUY":
                sell_sym, buy_sym = info.get("short_etf"), info["long_etf"]
            elif p["signal"] == "SHORT":
                sell_sym, buy_sym = info["long_etf"], info["short_etf"]
            else:
                sell_sym, buy_sym = info["long_etf"], None

            # Sell first
            if sell_sym and sell_sym in positions:
                qty = abs(float(positions[sell_sym].qty))
                if qty > 0:
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=sell_sym, qty=qty,
                        side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                    print(f"   ✅ SOLD {qty:.0f} {sell_sym}")
                    order_count += 1

            # Buy
            if buy_sym:
                trading_client.submit_order(MarketOrderRequest(
                    symbol=buy_sym, notional=round(per_ticker, 2),
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                print(f"   ✅ BUY ${per_ticker:,.0f} of {buy_sym} ({ticker})")
                order_count += 1

        except Exception as e:
            print(f"   ❌ {ticker}: {e}")

    print(f"   📊 {order_count} orders submitted")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🤖 AI DAILY STOCK PREDICTION v2 (37 features)")
    print("=" * 60)

    # 1. Download macro data
    print(f"\n⏳ Downloading macro data (VIX + cross-asset)...")
    macro_data = get_macro_data(years=6)
    print(f"   ✅ {macro_data.index[0].strftime('%Y-%m-%d')} → {macro_data.index[-1].strftime('%Y-%m-%d')}")

    # 2. Generate predictions
    print(f"\n⏳ Generating predictions...")
    predictions = {}
    for ticker in TICKERS:
        result = prepare_ticker(ticker, macro_data)
        predictions[ticker] = result
        emoji = "🟢" if result["signal"] == "BUY" else ("🔴" if result["signal"] == "SHORT" else "⚪")
        print(f"   {emoji} {ticker:<8} → {result['signal']:<6} ({result['confidence']:.1%}) → {result['trade']}")

    # 3. Print summary
    print(f"\n{'='*60}")
    print(f"  📋 TOMORROW 9:35 AM CHECKLIST")
    print(f"{'='*60}")
    for ticker in TICKERS:
        p = predictions[ticker]
        emoji = "🟢" if p["signal"] == "BUY" else ("🔴" if p["signal"] == "SHORT" else "⚪")
        print(f"  {emoji} {ticker:<8} → {p['trade']:<30} conf={p['confidence']:.1%}")

    # 4. Per-ticker detail
    print(f"\n{'='*60}")
    print(f"  📊 PER-TICKER DETAIL")
    print(f"{'='*60}")
    for ticker in TICKERS:
        p = predictions[ticker]
        print(f"\n── {ticker} ──")
        print(f"   Close: ${p['close']:.2f}  |  UP: {p['up_prob']:.1%}  |  DOWN: {p['down_prob']:.1%}")
        print(f"   Top 5 features:")
        for fname, fimp in p["top_5_features"]:
            val = p["current_values"].get(fname, 0)
            print(f"      {fname:<22} imp={fimp:.3f}  val={val:.4f}")

    # 5. Log to Google Sheets
    print(f"\n{'='*60}")
    print(f"  📝 GOOGLE SHEETS")
    print(f"{'='*60}")
    try:
        log_to_sheets(predictions)
    except Exception as e:
        print(f"   ❌ Sheets error: {e}")

    # 6. Auto-trade
    if AUTO_TRADE:
        try:
            execute_trades(predictions)
        except Exception as e:
            print(f"   ❌ Trading error: {e}")

    print(f"\n✅ Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
