"""Order execution via Alpaca."""


def target_symbols(signal, info):
    """Decide which symbol to hold and which to exit for a signal.

    Args:
        signal (str): ``"BUY"``, ``"SHORT"`` or ``"CASH"``.
        info (dict): Ticker info from :func:`common.tickers.classify_ticker`.

    Returns:
        tuple[str | None, str | None]: ``(desired, undesired)`` symbols.
        ``desired`` is None for CASH; ``undesired`` is None for a stock BUY.
    """
    if signal == "BUY":
        return info["long_etf"], info.get("short_etf")
    if signal == "SHORT":
        return info["short_etf"], info["long_etf"]
    return None, info["long_etf"]  # CASH


def execute_trades(predictions, ticker_info, tickers, api_key, secret_key, paper=True):
    """Submit market orders via Alpaca so each ticker is on the predicted side.

    Idempotent: positions already on the right side are left alone, and
    orders whose client order ID is already open are not resubmitted. For
    each ticker the undesired side is sold, then the desired side is bought
    with an equal share of portfolio value. Per-ticker errors are logged and
    do not stop other tickers.

    Args:
        predictions (dict[str, dict]): Predictions keyed by ticker (needs
            ``date`` and ``signal``).
        ticker_info (dict[str, dict]): Ticker info keyed by ticker.
        tickers (list[str]): Tickers to trade.
        api_key (str): Alpaca API key.
        secret_key (str): Alpaca secret key.
        paper (bool): True for paper trading, False for live.
    """
    if not api_key or not secret_key:
        print("   ❌ Missing Alpaca credentials")
        return

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = TradingClient(api_key, secret_key, paper=paper)
    account = client.get_account()
    mode = "🧪 PAPER" if paper else "💰 LIVE"
    print(f"\n   {mode} | Power: ${float(account.buying_power):,.2f} | Value: ${float(account.portfolio_value):,.2f}")

    positions = {p.symbol: p for p in client.get_all_positions()}
    # Check existing open orders to avoid duplicates
    open_order_ids = set()
    try:
        for o in client.get_orders():
            if getattr(o, "client_order_id", None):
                open_order_ids.add(o.client_order_id)
    except Exception as e:
        print(f"   ❌ Could not check open orders; skipping trading for safety: {e}")
        return
    per_ticker = float(account.portfolio_value) / len(tickers)
    orders = 0

    for ticker in tickers:
        p = predictions[ticker]
        date_str = p["date"]
        desired_sym, undesired_sym = target_symbols(p["signal"], ticker_info[ticker])

        try:
            # Sell undesired side if held
            if undesired_sym and undesired_sym in positions:
                qty = abs(float(positions[undesired_sym].qty))
                sell_cid = f"pred-close-{date_str}-{ticker}-{undesired_sym}"
                if qty > 0 and sell_cid not in open_order_ids:
                    client.submit_order(MarketOrderRequest(
                        symbol=undesired_sym, qty=qty,
                        side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        client_order_id=sell_cid,
                    ))
                    print(f"   ✅ SOLD {qty:.0f} {undesired_sym}")
                    orders += 1
                elif sell_cid in open_order_ids:
                    print(f"   ⏩ {ticker}: sell order already pending ({sell_cid})")

            if not desired_sym:
                continue
            # Skip buy if already holding the desired side
            if desired_sym in positions:
                print(f"   ⏩ {ticker}: already holding {desired_sym}")
                continue

            buy_cid = f"pred-open-{date_str}-{ticker}-{p['signal']}"
            if buy_cid in open_order_ids:
                print(f"   ⏩ {ticker}: order already pending ({buy_cid})")
                continue
            client.submit_order(MarketOrderRequest(
                symbol=desired_sym, notional=round(per_ticker, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                client_order_id=buy_cid,
            ))
            print(f"   ✅ BUY ${per_ticker:,.0f} {desired_sym}")
            orders += 1

        except Exception as e:
            print(f"   ❌ {ticker}: {e}")

    print(f"   📊 {orders} orders submitted")
