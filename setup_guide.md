# 🤖 Setup Guide — Daily Stock Prediction

## What This Does

Every weekday at 4:30 PM ET, GitHub Actions automatically:
1. Trains models with 37 features (including cross-asset signals)
2. Predicts tomorrow's direction for each ticker
3. Saves results to `predictions.csv` in your repo
4. Fills in past actuals (was the prediction correct?)
5. Shows a dashboard in the GitHub Actions run summary

## Setup (3 minutes)

### Step 1: Create a GitHub repo

Go to [github.com/new](https://github.com/new), name it `stock-predictions`, choose **Private**, click Create.

### Step 2: Upload the files

Your repo needs these 3 files:

```
stock-predictions/
├── .github/workflows/daily_prediction.yml
├── daily_predict.py
└── requirements.txt
```

- Upload `daily_predict.py` and `requirements.txt` via "Add file" → "Upload files"
- For the workflow: "Add file" → "Create new file" → type `.github/workflows/daily_prediction.yml` as the filename → paste its contents

### Step 3: Test it

1. Go to your repo → **Actions** tab
2. Click **"Daily Stock Prediction"** on the left
3. Click **"Run workflow"** → **"Run workflow"**
4. Watch it run (~3 min)
5. Check `predictions.csv` in your repo
6. Click the completed run → scroll down for the dashboard

**Done. It runs every weekday automatically.**

---

## Changing Config

All config is at the top of `daily_predict.py`. Edit and commit:

```python
TICKERS_INPUT     = "^GSPC, AAPL, NVDA, MSFT"   # change tickers here
DATA_SOURCE       = "yfinance"                    # or "alpaca"
ALPACA_API_KEY    = ""                            # fill in for Alpaca
ALPACA_SECRET_KEY = ""
ALPACA_PAPER      = True                          # False = live trading
ALPACA_FEED       = "iex"                         # "sip" for paid
AUTO_TRADE        = False                         # True = submit orders
```

> ⚠️ If you enable Alpaca with real keys, note that `daily_predict.py` is in your repo. Use a **private** repo to keep keys safe. Alternatively, move keys to GitHub Secrets and read them with `os.environ.get()`.

---

## Where to Find Results

**`predictions.csv`** — in your repo root. GitHub renders it as a table. Contains date, signal, confidence, actuals, and whether each prediction was correct.

**Job Summary** — click into any Actions run and scroll down. Shows a formatted dashboard with signals, cumulative accuracy, and top features.

**Logs** — click into any run for full output.

---

## Enabling Auto-Trading (Alpaca)

1. Edit `daily_predict.py`:
   - Set `DATA_SOURCE = "alpaca"`
   - Fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
   - Set `ALPACA_PAPER = True` (start with paper!)
   - Set `AUTO_TRADE = True`
2. Commit and push
3. Monitor paper trades for 2–4 weeks before considering live

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
