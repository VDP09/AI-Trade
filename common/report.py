"""Console output and GitHub Actions job summary."""

import os
from datetime import datetime

SIGNAL_EMOJI = {"BUY": "🟢", "SHORT": "🔴", "CASH": "⚪"}
LOW_CONFIDENCE = 0.55  # Confidence below this is flagged in the job summary


def signal_emoji(signal):
    """Look up the emoji for a signal.

    Args:
        signal (str): ``"BUY"``, ``"SHORT"`` or ``"CASH"``.

    Returns:
        str: The matching emoji (⚪ for unknown signals).
    """
    return SIGNAL_EMOJI.get(signal, "⚪")


def print_section(title):
    """Print a section title framed by ``=`` rules.

    Args:
        title (str): Section title.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_prediction(p):
    """Print a one-line summary of a prediction.

    Args:
        p (dict): Prediction from :func:`common.model.train_and_predict`.
    """
    print(f"   {signal_emoji(p['signal'])} {p['ticker']:<8} {p['signal']:<6} {p['confidence']:.1%}  → {p['trade']}")


def has_stats(stats):
    """Check whether any ticker has scored predictions.

    Args:
        stats (dict[str, dict]): Output of :func:`common.storage.compute_stats`.

    Returns:
        bool: True if at least one ticker has ``total > 0``.
    """
    return any(s["total"] > 0 for s in stats.values())


def print_stats(stats, tickers, unit):
    """Print cumulative accuracy and P&L per ticker (nothing if no stats yet).

    Args:
        stats (dict[str, dict]): Output of :func:`common.storage.compute_stats`.
        tickers (list[str]): Tickers to print, in order.
        unit (str): Label for the count column, e.g. ``"preds"`` or ``"weeks"``.
    """
    if not has_stats(stats):
        return
    print_section("📈 CUMULATIVE PERFORMANCE (directional model P&L, not exact trade P&L)")
    for ticker in tickers:
        s = stats[ticker]
        if s["total"] > 0:
            print(f"   {ticker:<8} {s['total']:>3} {unit} | Acc: {s['accuracy']:.1%} | P&L: {s['pnl']:+.2%}")


def print_checklist(predictions, tickers, title):
    """Print the trade action for each ticker.

    Args:
        predictions (dict[str, dict]): Predictions keyed by ticker.
        tickers (list[str]): Tickers to print, in order.
        title (str): Section title, e.g. ``"TOMORROW 9:35 AM"``.
    """
    print_section(f"📋 {title}")
    for ticker in tickers:
        p = predictions[ticker]
        print(f"   {signal_emoji(p['signal'])} {ticker:<8} → {p['trade']}")


def _table(headers):
    """Build the header and separator rows of a markdown table.

    Args:
        headers (list[str]): Column headers.

    Returns:
        list[str]: Two lines: the header row and the ``|---|`` separator.
    """
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|",
    ]


def write_job_summary(predictions, stats, tickers, title, signals_heading,
                      action_header="Action", count_header="Predictions"):
    """Append a markdown dashboard to the GitHub Actions job summary.

    Does nothing when ``GITHUB_STEP_SUMMARY`` is not set (e.g. running locally).

    Args:
        predictions (dict[str, dict]): Predictions keyed by ticker.
        stats (dict[str, dict]): Output of :func:`common.storage.compute_stats`.
        tickers (list[str]): Tickers to include, in order.
        title (str): Page title, e.g. ``"Daily Prediction"``.
        signals_heading (str): Heading above the signals table.
        action_header (str): Header for the trade-action column.
        count_header (str): Header for the prediction-count column.
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    today = datetime.now().strftime("%B %d, %Y")
    lines = [f"# 🤖 {title} — {today}", "", f"## {signals_heading}", ""]
    lines += _table(["Ticker", "Close", "Signal", action_header, "Strategy", "Confidence"])
    for ticker in tickers:
        p = predictions[ticker]
        sig = f"{signal_emoji(p['signal'])} {p['signal']}"
        lines.append(
            f"| **{ticker}** | ${p['close']:.2f} | {sig} | {p['trade']} | {p['strategy']} | {p['confidence']:.1%} |"
        )

    low_conf = [t for t in tickers if predictions[t]["confidence"] < LOW_CONFIDENCE]
    if low_conf:
        lines += ["", f"> ⚠️ **Low confidence**: {', '.join(low_conf)} — below {LOW_CONFIDENCE:.0%}"]

    if has_stats(stats):
        lines += ["", "## Cumulative Performance", ""]
        lines += _table(["Ticker", count_header, "Accuracy", "Cumulative P&L"])
        for ticker in tickers:
            s = stats[ticker]
            if s["total"] > 0:
                lines.append(f"| {ticker} | {s['total']} | {s['accuracy']:.1%} | {s['pnl']:+.2%} |")
            else:
                lines.append(f"| {ticker} | 0 | — | — |")

    lines += ["", "## Top Model Importance Features", ""]
    for ticker in tickers:
        top3 = ", ".join(f"`{n}` ({v:.3f})" for n, v in predictions[ticker]["top_5"][:3])
        lines.append(f"- **{ticker}**: {top3}")

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")
