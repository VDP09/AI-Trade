# AI-Trade — Design Document

> Purpose: a self-contained description of this project, detailed enough to be
> used as context when changing it. Diagrams: [`components.puml`](components.puml)
> (static structure) and [`sequence.puml`](sequence.puml) (one prediction run).

---

## 1. Overview

AI-Trade predicts the **direction** (up / down) of a small set of US stocks and
indices with an XGBoost classifier, logs each prediction to a CSV in the repo,
scores past predictions once the outcome is known, and optionally places the
matching trades through Alpaca.

There are two independent prediction jobs that share all their machinery:

| Job | Entry point | Horizon | Runs (GitHub Actions cron, UTC) | Log file | Auto-trade (default) |
|---|---|---|---|---|---|
| Daily | `predictors/daily_predict.py` | 1 trading day | Mon–Fri 21:30 and 22:00 (after US close) | `data/predictions.csv` | Off |
| Weekly | `predictors/weekly_predict.py` | 5 trading days | Sat 13:00 and 13:30 | `data/weekly_predictions.csv` | **On** (paper) |

Each workflow has two cron triggers 30 minutes apart; the second is a safety
net. Re-runs are harmless because CSV rows and Alpaca orders are de-duplicated
(see §8).

A static dashboard (`index.html`, served by GitHub Pages) reads both CSVs
straight from the `main` branch and charts accuracy and confidence.

**Everything runs for free:** GitHub Actions for compute and storage (the CSVs
are committed back to the repo), Yahoo Finance and Alpaca for data, Alpaca
paper trading for execution.

---

## 2. Repository layout

```
AI-Trade/
├── .github/workflows/
│   ├── daily_prediction.yml     # cron → python -m predictors.daily_predict → commit CSV
│   └── weekly_prediction.yml    # cron → python -m predictors.weekly_predict → commit CSV
├── config/                      # all settings (TOML), see §4
│   ├── common.toml              # shared: data source, Alpaca, index map, macro symbols
│   ├── daily.toml               # daily: tickers, horizon, features, XGB params, labels
│   └── weekly.toml              # weekly: same keys, weekly values
├── predictors/                  # entry points: feature set + macro features
│   ├── daily_predict.py
│   └── weekly_predict.py
├── common/                      # shared library (no job-specific constants)
│   ├── config.py                # load_config(): merge common.toml + <job>.toml
│   ├── pipeline.py              # PredictionJob + run(): orchestrates one run
│   ├── tickers.py               # ticker parsing, index/stock classification
│   ├── data.py                  # Yahoo / Alpaca downloaders, macro closes
│   ├── indicators.py            # technical indicators used by feature sets
│   ├── model.py                 # dataset, training, prediction → signal
│   ├── storage.py               # CSV log: load, back-fill outcomes, append, stats
│   ├── report.py                # console output + GitHub job summary
│   └── trading.py               # Alpaca order execution
├── data/                        # prediction logs (written by the workflows)
│   ├── predictions.csv
│   └── weekly_predictions.csv
├── design/                      # this document, diagrams, setup guide
├── index.html                   # GitHub Pages dashboard
├── requirements.txt
└── .flake8                      # lint + Google-style docstring checks
```

**Run from the repo root** with `python -m predictors.daily_predict` (or
`weekly_predict`). Running the file directly (`python predictors/daily_predict.py`)
fails to import `common`. `csv_file` paths in `config/` are resolved against
the repo root.

---

## 3. Architecture

The design separates **what differs per job** from **what is shared**:

- A predictor module loads its settings with `load_config("<name>")`, defines
  `get_macro_data(years)` and `create_features(df, macro_df)`, then builds a
  `PredictionJob` with `PredictionJob.from_config(...)` and calls
  `common.pipeline.run(job)`.
- All tunable values (tickers, data source, horizon, feature list, XGB params,
  index map, macro symbols, display labels) live in `config/*.toml` (§4).
- `common/` holds every reusable step. Modules depend only on each other via
  plain functions, DataFrames and dicts. There are no classes apart from the
  `PredictionJob` dataclass.

Dependency direction (no cycles):

```
predictors/*  →  common.pipeline  →  common.{data, model, storage, report, trading, tickers}
predictors/*  →  common.{indicators, data.load_macro_closes}
common.data   →  common.tickers
```

External dependencies: `yfinance`, `alpaca-py` (data + trading),
`xgboost`, `scikit-learn` (StandardScaler), `pandas`, `numpy`.
`alpaca` imports are lazy (inside functions), so the Yahoo-only path never
touches Alpaca.

---

## 4. Configuration

All settings live in TOML files under `config/` (parsed with the standard
library `tomllib`, so there is no extra dependency). `load_config(name)` in
`common/config.py` reads `config/common.toml`, then deep-merges
`config/<name>.toml` on top, so a predictor file can override any shared key
(e.g. `[trading] auto_trade`).

**`config/common.toml`** (shared):

| Key | Default | Meaning |
|---|---|---|
| `data.source` | `"alpaca"` | Price source for tickers: `"alpaca"` or `"yfinance"` (validated) |
| `data.history_years` | `5` | Years of price history downloaded per ticker for training |
| `data.backfill_years` | `1` | Years of price history downloaded to back-fill outcomes |
| `alpaca.paper` | `true` | Paper vs live trading |
| `alpaca.feed` | `"iex"` | `"iex"` (free) or `"sip"` (paid) |
| `trading.auto_trade` | `false` | Submit orders after predicting (Alpaca only) |
| `model.min_train_rows` | `252` | Minimum labelled rows needed to train |
| `macro.symbols` | VIX, SPY, TLT, UUP, GLD | Yahoo symbol → macro column name (§6.2) |
| `index_map` | 8 entries | Indices traded long/short (§5.1) |

**`config/daily.toml` / `config/weekly.toml`** (per predictor):

| Key | Daily | Weekly | Meaning |
|---|---|---|---|
| `tickers` | `["^GSPC", "AAPL", "NVDA", "MSFT"]` | same | Tickers to predict |
| `horizon` | `1` | `5` | Trading days ahead being predicted |
| `train_years` | `3` | `3` | Most recent years of labelled rows used to train |
| `macro_years` | `6` | `7` | Macro history downloaded |
| `csv_file` | `data/predictions.csv` | `data/weekly_predictions.csv` | Prediction log (relative to repo root) |
| `trading.auto_trade` | `false` | `true` | Overrides the shared default |
| `model.feature_cols` | 38 columns | 31 columns | Features fed to the model |
| `model.xgb_params` | see file | see file | `xgboost.XGBClassifier` keyword arguments |
| `labels.*` | see file | see file | Display text; `{n_features}` is substituted |

`model.feature_cols` can only list columns that the predictor's
`create_features` produces. `run()` raises a `ValueError` naming any that are
missing. Feature *definitions* (window lengths, formulas) stay in code.

Alpaca credentials are **never** in config files. `load_config` reads
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from the environment (GitHub Secrets in
CI) into `alpaca.api_key` / `alpaca.secret_key`.

`PredictionJob` (`common/pipeline.py`) carries these values plus the feature
function and macro function. Label keys: `banner`, `horizon_suffix`,
`predictions_name`, `stats_unit`, `checklist_title`, `summary_title`,
`signals_heading`, `action_header`, `count_header`.

---

## 5. Domain model

### 5.1 Tickers and strategies (`common/tickers.py`)

`[index_map]` in `config/common.toml` lists indices and index ETFs that can
be traded **long/short** via an inverse ETF. Anything else is a **stock**, traded **long/cash**.

| Ticker | Long ETF | Short (inverse) ETF | Alpaca data proxy |
|---|---|---|---|
| `^GSPC`, `SPY` | SPY | SH | SPY |
| `^DJI`, `DIA` | DIA | DOG | DIA |
| `^IXIC`, `QQQ` | QQQ | PSQ | QQQ |
| `^RUT`, `IWM` | IWM | RWM | IWM |

`classify_ticker()` returns an info dict:
`type` (`index`/`stock`), `strategy` (`long/short`/`long/cash`), `long_etf`,
`short_etf` (None for stocks), `name`, `up_action`, `down_action`.

### 5.2 Signals

| Model says | Index | Stock |
|---|---|---|
| UP (1) | `BUY` → "Buy SPY" | `BUY` → "Buy AAPL" |
| DOWN (0) | `SHORT` → "Buy SH" | `CASH` → "Sell AAPL → cash" |

### 5.3 Prediction record (in memory)

Returned by `common.model.train_and_predict()`:

```python
{
  "ticker": "AAPL", "date": "2026-10-02",      # date of the latest feature row
  "close": 255.46,                               # close on that date
  "signal": "BUY" | "SHORT" | "CASH", "trade": "Buy AAPL",
  "up_prob": 0.61, "down_prob": 0.39, "confidence": 0.61,   # confidence = max
  "top_feature": "Return_20d",
  "top_5": [("Return_20d", 0.052), ...],         # XGBoost feature importances
  "strategy": "long/cash",
}
```

---

## 6. Data

### 6.1 Price data (`common/data.py`)

- `make_downloader(data_source, ...)` returns a `download(ticker, years)` function.
- **Yahoo** (`download_data_yf`): `yf.download(..., auto_adjust=True)`, MultiIndex
  columns flattened, names title-cased.
- **Alpaca** (`make_alpaca_downloader`): daily bars via `StockHistoricalDataClient`;
  index tickers are fetched through their ETF proxy (`^GSPC` → SPY); the index is
  converted to tz-naive normalized dates named `Date`.
- The download window is `years * 365 + 60` days back from now (60-day buffer for
  rolling-window warm-up). Predictions download `data.history_years` (5);
  back-fill downloads `data.backfill_years` (1).
- Output is always an OHLCV DataFrame: `Open, High, Low, Close, Volume`.

> **Consequence:** with `data.source = "alpaca"`, the `Close` logged for `^GSPC`
> is the **SPY** price (≈ 1/10 of the index level), not the index itself.

### 6.2 Macro data

`load_macro_closes(years)` always uses **Yahoo** (Alpaca has no VIX): closes of
`^VIX`, `SPY`, `TLT` (bonds), `UUP` (dollar) and `GLD` (gold), outer-joined and
forward-filled. A missing series raises `RuntimeError`, which aborts the run.

Each predictor's `get_macro_data()` adds the derived columns it needs:

| Daily | Weekly |
|---|---|
| `VIX_SMA20`, `SPY_return_1d`, `SPY_return_5d`, `SPY_vs_SMA50`, `Bond_return_1d`, `Dollar_return_1d`, `Gold_return_1d` | `VIX_SMA20`, `VIX_weekly_chg`, `SPY_return_5d`, `SPY_return_20d`, `SPY_vs_SMA50`, `Bond_return_5d`, `Dollar_return_5d`, `Gold_return_5d` |

Macro columns are left-joined onto each ticker's dates and forward-filled
(`indicators.join_macro`) to cover differing holidays.

---

## 7. Features and model

### 7.1 Indicator library (`common/indicators.py`)

`ohlcv`, `atr_pct`, `close_position`, `bollinger_pct`, `up_volume_ratio`,
`obv_slope`, `rsi` (SMA variant), `macd_hist` (12/26/9, ÷ price),
`join_macro`, `add_vix_features`. All are price-normalized so features are
comparable across tickers.

### 7.2 Daily feature set: 38 columns

| Group | Features |
|---|---|
| Momentum (6) | `Return_1d/2d/3d/5d/20d`, `Overnight_gap` |
| Trend (2) | `SMA_10_50_ratio`, `Price_vs_SMA200` |
| Volatility (6) | `Volatility_5d/10d/20d`, `Vol_ratio_5_20`, `ATR_14`, `Intraday_range` |
| Price position (2) | `Close_pos_5d`, `BB_pct` |
| Volume (3) | `Volume_ratio`, `Up_volume_ratio` (10d), `OBV_slope` (10d) |
| Technical (3) | `RSI_14`, `Stochastic_K` (5d, 3-smoothed), `MACD_hist` |
| VIX (2) | `VIX`, `VIX_ratio` |
| Calendar (2) | `Day_sin`, `Day_cos` (day of week) |
| Market regime (3) | `SPY_return_1d`, `SPY_return_5d`, `SPY_vs_SMA50` |
| Cross-asset (3) | `Bond_return_1d`, `Dollar_return_1d`, `Gold_return_1d` |
| Momentum dynamics (3) | `RSI_3d_change`, `MACD_accel`, `Vol_accel` |
| Lagged (3) | `Volume_ratio_lag1`, `Overnight_gap_lag1`, `Return_rank_5d` |

### 7.3 Weekly feature set: 31 columns

Tuned for a 5-day horizon: no ultra-short features, wider windows, 5-day
cross-asset returns, no stochastic.

| Group | Features |
|---|---|
| Momentum (5) | `Return_5d/10d/20d/60d`, `Return_5d_lag1` |
| Trend (3) | `SMA_10_50_ratio`, `SMA_50_200_ratio`, `Price_vs_SMA200` |
| Volatility (4) | `Volatility_20d/60d`, `Vol_ratio_20_60`, `ATR_14` |
| Price position (2) | `Close_pos_20d`, `BB_pct` |
| Volume (3) | `Volume_ratio`, `Up_volume_ratio` (20d), `OBV_slope` (20d) |
| Technical (3) | `RSI_14`, `MACD_hist`, `RSI_5d_change` |
| VIX (3) | `VIX`, `VIX_ratio`, `VIX_weekly_chg` |
| Market regime (3) | `SPY_return_5d`, `SPY_return_20d`, `SPY_vs_SMA50` |
| Cross-asset (3) | `Bond_return_5d`, `Dollar_return_5d`, `Gold_return_5d` |
| Momentum dynamics (2) | `MACD_accel_5d`, `Vol_accel` |

### 7.4 Training (`common/model.py`)

A fresh model is trained **per ticker, per run**. Nothing is persisted.

1. `build_dataset`: target = 1 if `Close[t + horizon] > Close[t]`, else 0.
   Rows whose future is unknown keep **NaN** (never cast to 0, which would be a
   false DOWN label). Rows with any NaN feature are dropped.
2. Drop the last `horizon` rows and any NaN-target rows; require ≥ 252 rows
   (`model.min_train_rows`), else `ValueError` (aborts the run).
3. Keep the most recent `train_years * 252` rows.
4. `StandardScaler` fit on the training rows only, then `XGBClassifier.fit`.
5. Predict the **latest** row (the most recent close): class, probabilities,
   and feature importances.

| XGB param | Daily | Weekly |
|---|---|---|
| `n_estimators` | 400 | 250 |
| `max_depth` | 4 | 4 |
| `learning_rate` | 0.02 | 0.04 |
| `subsample` / `colsample_bytree` | 0.7 / 0.6 | 0.8 / 0.7 |
| `reg_alpha` / `reg_lambda` | 3.0 / 4.0 | 1.5 / 2.5 |
| `min_child_weight` / `gamma` | 8 / 0.15 | 5 / 0.1 |
| `eval_metric`, `random_state` | logloss, 42 | logloss, 42 |

Daily is regularized more heavily because 1-day returns are noisier.

---

## 8. Prediction log (`common/storage.py`)

### 8.1 Schema

All values are stored as strings; a blank field means "not known yet".

| Column | Written when | Example |
|---|---|---|
| `Date` | prediction | `2026-10-02` (date of the close the prediction is made from) |
| `Ticker` | prediction | `^GSPC` |
| `Close` | prediction | `669.21` |
| `Signal` | prediction | `BUY` / `SHORT` / `CASH` |
| `Trade` | prediction | `Buy SPY` |
| `Strategy` | prediction | `long/short` / `long/cash` |
| `Confidence`, `UP_Prob`, `DOWN_Prob` | prediction | `0.6123` |
| `Actual_Direction` | back-fill | `UP` / `DOWN` / `FLAT` |
| `Actual_Close` | back-fill | close `horizon` trading rows later |
| `Market_Return` | back-fill | `(actual - close) / close` |
| `Strategy_Return` | back-fill | BUY: market return; SHORT: negative of it; CASH: 0 |
| `Correct` | back-fill | `Yes` / `No` (CASH is correct unless UP) |
| `Top_Feature` | prediction | most important feature |

### 8.2 Behaviour

- **Load:** `keep_default_na=False` so blanks stay `""`; missing columns are
  added. A missing file starts an empty log. The `data/` folder is created
  on save if needed.
- **Back-fill:** for each unfilled row at least `horizon` business days old,
  download the ticker once (cached per run) and take the close `horizon`
  trading rows after `Date`. Errors are logged per row and skipped.
- **Append:** one row per ticker, skipped if `(Date, Ticker)` already exists.
  This makes re-runs and the second cron trigger idempotent.
- **Stats:** per ticker, over rows with `Correct` ∈ {Yes, No}: accuracy and the
  **sum** of `Strategy_Return` (a directional, uncompounded P&L, not exact
  trade P&L).

---

## 9. Trading (`common/trading.py`)

Runs only when `auto_trade` is on **and** `data_source == "alpaca"`, and only
with credentials.

1. Read the account (buying power, portfolio value), positions and open orders.
   If open orders can't be read, **skip all trading** (safety).
2. Budget per ticker = portfolio value ÷ number of tickers.
3. For each ticker, `target_symbols(signal, info)` gives `(desired, undesired)`:
   - BUY → hold long ETF/stock, exit short ETF.
   - SHORT → hold inverse ETF, exit long ETF.
   - CASH → hold nothing, exit the stock.
4. Sell the full position in `undesired` if held (market, DAY).
5. Buy `desired` for the notional budget unless already held (market, DAY).

**Idempotency:** each order has a deterministic `client_order_id`:
`pred-close-{date}-{ticker}-{symbol}` (sell) and
`pred-open-{date}-{ticker}-{signal}` (buy). Orders whose ID is already open are
not resubmitted, and existing positions on the right side are left alone.
Per-ticker errors are logged and don't stop other tickers.

The weekly job runs on Saturday, so its DAY market orders queue for Monday's
open, which is why its checklist says "MONDAY 9:35 AM".

---

## 10. Reporting

- **Console** (`common/report.py`): banner, per-ticker prediction line,
  cumulative stats, and a trade checklist.
- **GitHub job summary:** markdown appended to `$GITHUB_STEP_SUMMARY` (no-op
  locally): signals table, low-confidence warning (< 55%), cumulative
  performance table, and top-3 features per ticker.
- **Dashboard** (`index.html`): static page using PapaParse and Chart.js from
  cdnjs. It fetches
  `https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/data/*.csv`
  (`GITHUB_USER`/`GITHUB_REPO` are edited at the top of the script). It has
  Daily/Weekly tabs, each with the latest signals, cumulative performance,
  rolling accuracy and confidence-over-time charts, and the prediction
  history table. Because the CSVs are read from `main`, the dashboard updates
  as soon as a workflow commits.

---

## 11. Run flow

See [`sequence.puml`](sequence.puml). `common.pipeline.run(job)`:

1. Create the downloader; parse and classify tickers; print the banner.
2. `get_macro_data(macro_years)`. This raises if macro data is missing.
3. For each ticker: download 5 years → `create_features` → `train_and_predict`.
   This raises if history is too short.
4. Load the CSV; back-fill matured rows.
5. Append the new predictions; write the CSV.
6. Compute and print stats; print the checklist.
7. Write the GitHub job summary.
8. If enabled, execute trades. Exceptions here are caught and logged, and the
   run still succeeds.

The workflow then runs `git add data/<file>.csv`, commits only if it changed
(`📊 Predictions YYYY-MM-DD` / `📊 Weekly predictions YYYY-MM-DD`), and pushes
using the job's `contents: write` permission.

---

## 12. Failure modes and safety

| Situation | Behaviour |
|---|---|
| Macro series missing | `RuntimeError`, run fails, nothing written |
| Ticker history < 252 trainable rows | `ValueError`, run fails, nothing written |
| Back-fill download error | Logged, row left blank, retried next run |
| Workflow runs twice / manual re-run | Duplicate rows and orders skipped |
| Open orders unreadable | All trading skipped |
| Single order fails | Logged; other tickers continue |
| Missing Alpaca keys with `data.source = "alpaca"` | Current alpaca-py rejects an unauthenticated client, so the run fails at startup |
| Not in GitHub Actions | Job summary skipped |

---

## 13. Known limitations

- **Index close under Alpaca** is the ETF proxy price (§6.1). Direction and
  returns are still consistent within a log, but the values aren't index levels.
- **Back-fill window:** back-fill downloads ~14 months. A row left unfilled for
  longer than that would be scored against the wrong close.
- **Business-day gate** (`np.busday_count`) ignores market holidays. It only
  decides when to try; the actual outcome uses real trading rows.
- **P&L** is the sum of simple returns, uncompounded, with no costs or slippage.
- **Shared Alpaca account:** daily and weekly would trade the same symbols with
  separate budgets if both had `auto_trade` on. Only weekly does by default.
- **No model persistence or validation set:** the model is retrained every run;
  accuracy is only measured live through the log.
- Daily and weekly tickers are configured separately and can drift apart.

---

## 14. Development

- Install: `pip install -r requirements.txt`.
- Run locally from the repo root: `python -m predictors.daily_predict`. This
  writes to `data/`, so revert the CSV if you don't want the test rows.
- Lint: `pip install flake8 flake8-docstrings && flake8` (config in `.flake8`:
  max line 120, Google docstring convention).
- Conventions: Google-style docstrings with `Args`/`Returns`/`Raises`;
  emoji-prefixed console output; aligned config blocks are intentional
  (E221/E241 ignored).

### Extension points

| Change | Where |
|---|---|
| Add/remove tickers | `tickers` in `config/<job>.toml` |
| Tune the model | `[model.xgb_params]` in `config/<job>.toml` |
| Drop a feature | Remove it from `model.feature_cols` in `config/<job>.toml` |
| New index with an inverse ETF | `[index_map]` in `config/common.toml` |
| New feature | `create_features` + `model.feature_cols` (+ a helper in `indicators.py` if reusable) |
| New macro series | `[macro.symbols]` in `config/common.toml` + derived column in `get_macro_data` |
| New horizon (e.g. monthly) | New `config/<job>.toml` + predictor module + workflow; reuse `common.pipeline.run` |
| Different model | `common/model.py: train_and_predict` (keep the returned dict shape) |
| New data source | Add a downloader returning OHLCV and wire it in `make_downloader` |
