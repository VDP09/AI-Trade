"""End-to-end prediction run shared by the daily and weekly scripts."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from common.data import make_downloader
from common.model import train_and_predict
from common.report import (print_checklist, print_prediction, print_section,
                           print_stats, write_job_summary)
from common.storage import compute_stats, fill_past_actuals, load_csv, save_predictions
from common.tickers import build_ticker_info, parse_tickers
from common.trading import execute_trades


@dataclass
class PredictionJob:
    """Everything that differs between the daily and weekly prediction runs.

    Usually built from the TOML config with :meth:`from_config`.

    Attributes:
        tickers (list[str] | str): Tickers, e.g. ``["^GSPC", "AAPL"]`` or the
            comma-separated string ``"^GSPC, AAPL"``.
        horizon (int): Trading days ahead to predict.
        train_years (int): Years of history to train on.
        csv_file (pathlib.Path): Prediction log location.
        feature_cols (list[str]): Feature columns used by the model.
        xgb_params (dict): Keyword arguments for ``xgboost.XGBClassifier``.
        create_features (Callable): ``(price_df, macro_df) -> DataFrame`` of
            features.
        get_macro_data (Callable): ``(years) -> DataFrame`` of macro series.
        macro_years (int): Years of macro history to download.
        index_map (dict[str, dict]): Indices traded long/short; see
            :func:`common.tickers.classify_ticker`.
        min_train_rows (int): Minimum labelled rows needed to train.
        history_years (int): Years of price history downloaded per ticker.
        backfill_years (int): Years of price history downloaded to fill in
            past outcomes.
        data_source (str): ``"alpaca"`` or ``"yfinance"``.
        alpaca_api_key (str): Alpaca API key.
        alpaca_secret_key (str): Alpaca secret key.
        alpaca_paper (bool): True for paper trading, False for live.
        alpaca_feed (str): Alpaca data feed, ``"iex"`` or ``"sip"``.
        auto_trade (bool): Submit orders via Alpaca after predicting.
        labels (dict[str, str]): Text for console output and the job summary.
            Keys: ``banner``, ``horizon_suffix``, ``predictions_name``,
            ``stats_unit``, ``checklist_title``, ``summary_title``,
            ``signals_heading``, ``action_header``, ``count_header``.
    """

    # Model
    tickers: list
    horizon: int
    train_years: int
    csv_file: Path
    feature_cols: list
    xgb_params: dict
    create_features: Callable
    get_macro_data: Callable
    macro_years: int
    index_map: dict = field(default_factory=dict)
    min_train_rows: int = 252
    history_years: int = 5
    backfill_years: int = 1

    # Data source / trading
    data_source: str = "alpaca"
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    alpaca_feed: str = "iex"
    auto_trade: bool = False

    # Labels for console + job summary
    labels: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg, create_features, get_macro_data):
        """Build a job from settings returned by :func:`common.config.load_config`.

        Args:
            cfg (dict): Merged predictor settings.
            create_features (Callable): The predictor's feature function.
            get_macro_data (Callable): The predictor's macro function.

        Returns:
            PredictionJob: Job ready to pass to :func:`run`.
        """
        model = cfg["model"]
        feature_cols = list(model["feature_cols"])
        labels = {k: v.format(n_features=len(feature_cols)) for k, v in cfg.get("labels", {}).items()}
        return cls(
            tickers=cfg["tickers"],
            horizon=cfg["horizon"],
            train_years=cfg["train_years"],
            csv_file=Path(cfg["csv_file"]),
            feature_cols=feature_cols,
            xgb_params=dict(model["xgb_params"]),
            create_features=create_features,
            get_macro_data=get_macro_data,
            macro_years=cfg["macro_years"],
            index_map=cfg["index_map"],
            min_train_rows=model["min_train_rows"],
            history_years=cfg["data"]["history_years"],
            backfill_years=cfg["data"]["backfill_years"],
            data_source=cfg["data"]["source"],
            alpaca_api_key=cfg["alpaca"]["api_key"],
            alpaca_secret_key=cfg["alpaca"]["secret_key"],
            alpaca_paper=cfg["alpaca"]["paper"],
            alpaca_feed=cfg["alpaca"]["feed"],
            auto_trade=cfg["trading"]["auto_trade"],
            labels=labels,
        )

    def label(self, key, default):
        """Look up a display label.

        Args:
            key (str): Label key, e.g. ``"banner"``.
            default (str): Value to use when the label is not set.

        Returns:
            str: The configured label, or ``default``.
        """
        return self.labels.get(key, default)


def run(job: PredictionJob):
    """Run a full prediction cycle.

    Steps: download macro data, train and predict each ticker, back-fill
    past outcomes in the CSV log, append new predictions, print stats and
    the trade checklist, write the GitHub Actions job summary, and (if
    enabled with Alpaca) submit trades.

    Args:
        job (PredictionJob): Configuration for this run.

    Raises:
        RuntimeError: If macro data cannot be downloaded.
        ValueError: If a ticker has too little history to train on, or a
            configured feature column is not produced by ``create_features``.
    """
    download_data = make_downloader(job.data_source, job.alpaca_api_key,
                                    job.alpaca_secret_key, job.alpaca_feed, job.index_map)
    tickers = parse_tickers(job.tickers)
    ticker_info = build_ticker_info(tickers, job.index_map)

    print("\n" + "=" * 60)
    print(f"  🤖 {job.label('banner', 'AI STOCK PREDICTION')}")
    print("=" * 60)
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Features: {len(job.feature_cols)} | Horizon: {job.horizon}d{job.label('horizon_suffix', '')}")

    # 1. Macro data
    print("\n⏳ Downloading macro data...")
    macro_data = job.get_macro_data(years=job.macro_years)
    print(f"   ✅ {macro_data.index[0].strftime('%Y-%m-%d')} → {macro_data.index[-1].strftime('%Y-%m-%d')}")

    # 2. Predictions
    print(f"\n⏳ Generating {job.label('predictions_name', 'predictions')}...")
    predictions = {}
    for ticker in tickers:
        df = download_data(ticker, years=job.history_years)
        feat = job.create_features(df, macro_data)
        missing = [c for c in job.feature_cols if c not in feat.columns]
        if missing:
            raise ValueError(f"Configured feature_cols not produced by create_features: {missing}")
        predictions[ticker] = train_and_predict(
            ticker, df, feat, ticker_info[ticker],
            job.feature_cols, job.xgb_params, job.horizon, job.train_years,
            job.min_train_rows,
        )
        print_prediction(predictions[ticker])

    # 3. CSV: fill past actuals
    print(f"\n⏳ Updating {job.csv_file}...")
    log = load_csv(job.csv_file)
    log, filled = fill_past_actuals(log, job.horizon, download_data, job.backfill_years)
    if filled:
        print(f"   ✅ Filled {filled} past result(s)")

    # 4. CSV: save new predictions
    log, added = save_predictions(log, predictions, tickers)
    job.csv_file.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(job.csv_file, index=False)
    print(f"   📊 Added {added} prediction(s) | Total rows: {len(log)}")

    # 5. Stats
    stats = compute_stats(log, tickers)
    print_stats(stats, tickers, unit=job.label("stats_unit", "preds"))

    # 6. Checklist
    print_checklist(predictions, tickers, job.label("checklist_title", "TRADE CHECKLIST"))

    # 7. GitHub Actions summary
    write_job_summary(
        predictions, stats, tickers,
        title=job.label("summary_title", "Prediction"),
        signals_heading=job.label("signals_heading", "Signals"),
        action_header=job.label("action_header", "Action"),
        count_header=job.label("count_header", "Predictions"),
    )

    # 8. Auto-trade
    if job.auto_trade and job.data_source == "alpaca":
        print_section("🤖 AUTO-TRADING")
        try:
            execute_trades(predictions, ticker_info, tickers,
                           job.alpaca_api_key, job.alpaca_secret_key, job.alpaca_paper)
        except Exception as e:
            print(f"   ❌ {e}")

    print(f"\n✅ Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
