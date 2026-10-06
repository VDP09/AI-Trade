# 🤖 Setup Guide — Daily Stock Prediction

## What This Does

Every weekday at 4:30 PM ET, GitHub Actions automatically:
1. Trains models with 37 features (including cross-asset signals)
2. Predicts tomorrow's direction for each ticker
3. Saves results to `data/predictions.csv` in your repo
4. Fills in past actuals (was the prediction correct?)
5. Shows a dashboard in the GitHub Actions run summary

## Setup (3 minutes)

### Step 1: Create a GitHub repo

Go to [github.com/new](https://github.com/new), name it `stock-predictions`, choose **Private**, click Create.

### Step 2: Upload the files

Your repo needs these files:

```
stock-predictions/
├── .github/workflows/daily_prediction.yml
├── predictors/
│   ├── __init__.py
│   └── daily_predict.py   # run with: python -m predictors.daily_predict
├── common/                # shared modules used by the predictors
├── data/                  # prediction CSVs (created/updated by the workflows)
└── requirements.txt
```

- Upload the `predictors/` and `common/` folders and `requirements.txt` via "Add file" → "Upload files"
- For the workflow: "Add file" → "Create new file" → type `.github/workflows/daily_prediction.yml` as the filename → paste its contents

### Step 3: Test it

1. Go to your repo → **Actions** tab
2. Click **"Daily Stock Prediction"** on the left
3. Click **"Run workflow"** → **"Run workflow"**
4. Watch it run (~3 min)
5. Check `data/predictions.csv` in your repo
6. Click the completed run → scroll down for the dashboard

**Done. It runs every weekday automatically.**

---

## Changing Config

All settings live in TOML files under `config/`. Edit and commit:

- `config/common.toml` holds shared settings: data source, Alpaca paper/live and feed, the index → ETF map, and macro symbols.
- `config/daily.toml` and `config/weekly.toml` hold per-predictor settings: tickers, horizon, auto-trade, feature list, XGBoost params and display labels. Any key from `common.toml` can be overridden here.

```toml
# config/common.toml
[data]
source = "yfinance"        # or "alpaca"

[alpaca]
paper = true               # false = live trading
feed  = "iex"              # "sip" for paid

# config/daily.toml
tickers = ["^GSPC", "AAPL", "NVDA", "MSFT"]   # change tickers here

[trading]
auto_trade = false         # true = submit orders
```

> 🔐 Alpaca keys are **never** stored in config files. They are read from the `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` environment variables, which the workflows fill from GitHub Secrets.

---

## Where to Find Results

**`data/predictions.csv`** — in the `data/` folder of your repo. GitHub renders it as a table. Contains date, signal, confidence, actuals, and whether each prediction was correct.

**Job Summary** — click into any Actions run and scroll down. Shows a formatted dashboard with signals, cumulative accuracy, and top features.

**Logs** — click into any run for full output.

---

## Enabling Auto-Trading (Alpaca)

1. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` as GitHub Secrets (Settings → Secrets and variables → Actions)
2. In `config/common.toml`, set `source = "alpaca"` under `[data]` and keep `paper = true` under `[alpaca]` (start with paper!)
3. In `config/daily.toml` (or `weekly.toml`), set `auto_trade = true` under `[trading]`
4. Commit and push
5. Monitor paper trades for 2–4 weeks before considering live

---

## Cost: $0/month

| Component | Cost |
|---|---|
| GitHub Actions (private repo) | Free (uses ~66 of 2,000 free min/month) |
| Yahoo Finance data | Free |
| Alpaca data + trading | Free |

---

## FAQ

**How do I stop it?** Actions tab → "Daily Stock Prediction" → "..." menu → "Disable workflow"

**What if it misses a day?** Click "Run workflow" manually. It skips duplicates, so re-running is safe.

**Can I still use the Colab notebook?** Yes — the notebook logs to Google Sheets, this logs to CSV. They're independent.
