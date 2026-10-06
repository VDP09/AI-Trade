"""Ticker parsing and classification.

Indices (and their ETFs) are traded long/short via a pair of ETFs; individual
stocks are traded long/cash. Which tickers count as indices is configured in
``[index_map]`` in ``config/common.toml``: each entry maps a ticker to
``{"long", "short", "name", "alpaca_proxy"}``.
"""


def parse_tickers(tickers_input):
    """Normalize a ticker list.

    Args:
        tickers_input (list[str] | str): Tickers as a list, or a
            comma-separated string, e.g. ``"^GSPC, aapl"``.

    Returns:
        list[str]: Upper-cased, whitespace-stripped tickers with blanks
        removed, e.g. ``["^GSPC", "AAPL"]``.
    """
    if isinstance(tickers_input, str):
        tickers_input = tickers_input.split(",")
    return [t.strip().upper() for t in tickers_input if t.strip()]


def classify_ticker(ticker, index_map):
    """Describe how a ticker is traded.

    Args:
        ticker (str): Ticker symbol, e.g. ``"^GSPC"`` or ``"AAPL"``.
        index_map (dict[str, dict]): Indices traded long/short.

    Returns:
        dict: Trading info with keys ``type`` ("index" or "stock"),
        ``strategy`` ("long/short" or "long/cash"), ``long_etf``,
        ``short_etf`` (None for stocks), ``name``, ``up_action`` and
        ``down_action`` (human-readable trade instructions).
    """
    if ticker in index_map:
        m = index_map[ticker]
        return {
            "type": "index", "strategy": "long/short",
            "long_etf": m["long"], "short_etf": m["short"],
            "name": m["name"],
            "up_action": f"Buy {m['long']}",
            "down_action": f"Buy {m['short']}",
        }
    return {
        "type": "stock", "strategy": "long/cash",
        "long_etf": ticker, "short_etf": None, "name": ticker,
        "up_action": f"Buy {ticker}", "down_action": f"Sell {ticker} → cash",
    }


def build_ticker_info(tickers, index_map):
    """Classify every ticker in a list.

    Args:
        tickers (list[str]): Ticker symbols.
        index_map (dict[str, dict]): Indices traded long/short.

    Returns:
        dict[str, dict]: Mapping of ticker to its :func:`classify_ticker` info.
    """
    return {t: classify_ticker(t, index_map) for t in tickers}


def alpaca_symbol(ticker, index_map):
    """Map a ticker to the symbol Alpaca can serve data for.

    Alpaca has no index data, so indices are mapped to their ETF proxy.

    Args:
        ticker (str): Ticker symbol, e.g. ``"^GSPC"``.
        index_map (dict[str, dict]): Indices traded long/short.

    Returns:
        str: The ETF proxy for indices (e.g. ``"SPY"``), otherwise the ticker
        unchanged.
    """
    return index_map[ticker]["alpaca_proxy"] if ticker in index_map else ticker
