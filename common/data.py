"""Price data download (Yahoo Finance / Alpaca) and macro series.

All downloaders return a daily OHLCV DataFrame indexed by date with the
columns ``Open``, ``High``, ``Low``, ``Close`` and ``Volume``.
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from common.tickers import alpaca_symbol


def _date_range(years):
    """Compute the download window ending now.

    Args:
        years (int): Number of years of history; 60 extra days are added as a
            warm-up buffer for rolling indicators.

    Returns:
        tuple[datetime, datetime]: ``(start, end)``.
    """
    end = datetime.now()
    return end - timedelta(days=years * 365 + 60), end


def download_data_yf(ticker, years=5):
    """Download daily price history from Yahoo Finance.

    Args:
        ticker (str): Yahoo ticker, e.g. ``"^GSPC"`` or ``"AAPL"``.
        years (int): Years of history to download.

    Returns:
        pandas.DataFrame: Daily OHLCV data (split/dividend adjusted) indexed
        by date, with title-cased column names.
    """
    start, end = _date_range(years)
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


def make_alpaca_downloader(api_key="", secret_key="", feed="iex", index_map=None):
    """Create an Alpaca historical-data client and a downloader bound to it.

    Args:
        api_key (str): Alpaca API key. If empty (or ``secret_key`` is empty),
            an unauthenticated client is created.
        secret_key (str): Alpaca secret key.
        feed (str): ``"iex"`` (free tier) or ``"sip"`` (paid).
        index_map (dict[str, dict] | None): Indices traded long/short, used to
            map indices to their ETF proxy.

    Returns:
        Callable[[str, int], pandas.DataFrame]: A ``download(ticker, years)``
        function returning daily OHLCV data. Indices are fetched via their
        ETF proxy (see :func:`common.tickers.alpaca_symbol`).
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    index_map = index_map or {}
    data_feed = DataFeed.SIP if feed == "sip" else DataFeed.IEX
    if api_key and secret_key:
        client = StockHistoricalDataClient(api_key, secret_key)
        print(f"📡 Alpaca (feed={feed})")
    else:
        client = StockHistoricalDataClient()
        print("📡 Alpaca (unauthenticated)")

    def download_data_alpaca(ticker, years=5):
        """Download daily bars for one ticker from Alpaca.

        Args:
            ticker (str): Ticker symbol; indices are mapped to an ETF proxy.
            years (int): Years of history to download.

        Returns:
            pandas.DataFrame: Daily OHLCV data indexed by tz-naive,
            normalized dates (index name ``"Date"``).
        """
        start, end = _date_range(years)
        request = StockBarsRequest(
            symbol_or_symbols=alpaca_symbol(ticker, index_map), timeframe=TimeFrame.Day,
            start=start, end=end, feed=data_feed,
        )
        df = client.get_stock_bars(request).df
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

    return download_data_alpaca


def make_downloader(data_source, api_key="", secret_key="", feed="iex", index_map=None):
    """Pick the price downloader for the configured data source.

    Args:
        data_source (str): ``"alpaca"`` or ``"yfinance"``.
        api_key (str): Alpaca API key (Alpaca only).
        secret_key (str): Alpaca secret key (Alpaca only).
        feed (str): Alpaca data feed, ``"iex"`` or ``"sip"`` (Alpaca only).
        index_map (dict[str, dict] | None): Indices traded long/short, used to
            map indices to their ETF proxy (Alpaca only).

    Returns:
        Callable[[str, int], pandas.DataFrame]: A ``download(ticker, years)``
        function returning daily OHLCV data.
    """
    if data_source == "alpaca":
        return make_alpaca_downloader(api_key, secret_key, feed, index_map)
    print("📡 Yahoo Finance")
    return download_data_yf


def load_macro_closes(years, symbols):
    """Download VIX and cross-asset closing prices (always via Yahoo Finance).

    Args:
        years (int): Years of history to download.
        symbols (dict[str, str]): Yahoo symbol → column name, from
            ``[macro.symbols]`` in ``config/common.toml``.

    Returns:
        pandas.DataFrame: Forward-filled closes, one column per entry in
        ``symbols`` (e.g. ``VIX``, ``SPY``, ``TLT``, ``UUP``, ``GLD``),
        indexed by date.

    Raises:
        RuntimeError: If any required series failed to download or the
            result is empty.
    """
    macro = pd.DataFrame()
    for sym, label in symbols.items():
        print(f"   📥 {label}...")
        try:
            close = download_data_yf(sym, years)["Close"].rename(label)
            macro = macro.join(close, how="outer") if len(macro) > 0 else pd.DataFrame(close)
        except Exception as e:
            print(f"      ⚠️ {sym}: {e}")

    missing = set(symbols.values()) - set(macro.columns)
    if missing:
        raise RuntimeError(f"Missing required macro data: {sorted(missing)}. Check network/API.")
    if macro.empty:
        raise RuntimeError("Macro data frame is empty. Cannot proceed.")
    return macro.ffill()
