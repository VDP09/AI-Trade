"""Shared building blocks for the daily and weekly prediction scripts.

Modules:
    tickers: Ticker parsing and index/stock classification.
    data: Price and macro data download (Yahoo Finance / Alpaca).
    indicators: Technical indicator helpers used to build features.
    model: XGBoost training and signal generation.
    storage: Prediction CSV log (load, back-fill outcomes, append, stats).
    report: Console output and GitHub Actions job summary.
    trading: Order execution via Alpaca.
    pipeline: End-to-end run that ties the modules together.
"""
