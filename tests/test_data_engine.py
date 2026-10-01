"""Average-cost profit, fees, oversell checks and missing prices in data_engine."""

from datetime import date

import pandas as pd
import pytest

import data_engine as de


@pytest.fixture
def conn(tmp_path):
    c = de.init_db(tmp_path / "test.db", seed_demo=False)
    yield c
    c.close()


def prices(btc=None, gold=None, idr=None):
    return {"BTC_USD": btc, "GOLD_USD_per_gram": gold, "IDR_per_USD": idr}


def test_partial_sale_uses_average_cost(conn):
    # The case the old code got wrong: it showed a $20,000 average cost and $20,000 profit.
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "SELL", 0.5, 60_000, ts=date(2025, 2, 1))
    btc = de.compute_portfolio(conn, prices(btc=60_000, gold=100, idr=16_000))["BTC"]
    assert btc["qty"] == pytest.approx(0.5)
    assert btc["avg_cost"] == pytest.approx(40_000)
    assert btc["realised_pnl_usd"] == pytest.approx(10_000)
    assert btc["unrealised_pnl_usd"] == pytest.approx(10_000)
    assert btc["pnl_usd"] == pytest.approx(20_000)
    assert btc["pnl_pct"] == pytest.approx(50.0)  # $20,000 profit on $40,000 put in
    assert btc["pnl_idr"] == pytest.approx(20_000 * 16_000)


def test_buy_fee_is_part_of_the_cost(conn):
    # The old code stored the fee and never used it: a $250 fee left the profit at $0.
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, fee_usd=250, ts=date(2025, 1, 1))
    btc = de.compute_portfolio(conn, prices(btc=40_000, gold=100, idr=16_000))["BTC"]
    assert btc["avg_cost"] == pytest.approx(40_250)
    assert btc["pnl_usd"] == pytest.approx(-250)


def test_sell_fee_reduces_realised_profit(conn):
    de.add_transaction(conn, "GOLD", "BUY", 100, 60, ts=date(2025, 1, 1))
    de.add_transaction(conn, "GOLD", "SELL", 40, 70, fee_usd=8, ts=date(2025, 3, 1))
    gold = de.compute_portfolio(conn, prices(btc=1, gold=70, idr=16_000))["GOLD"]
    assert gold["realised_pnl_usd"] == pytest.approx(40 * 70 - 8 - 40 * 60)
    assert gold["unrealised_pnl_usd"] == pytest.approx(60 * 70 - 60 * 60)
    assert gold["fees"] == pytest.approx(8)


def test_several_buys_then_selling_out(conn):
    de.add_transaction(conn, "BTC", "BUY", 0.05, 42_000, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "BUY", 0.03, 38_500, ts=date(2025, 2, 1))
    de.add_transaction(conn, "BTC", "BUY", 0.02, 55_000, ts=date(2025, 3, 1))
    btc = de.compute_portfolio(conn, prices(btc=50_000, gold=1, idr=1))["BTC"]
    assert btc["avg_cost"] == pytest.approx(43_550)
    de.add_transaction(conn, "BTC", "SELL", 0.1, 50_000, ts=date(2025, 4, 1))
    btc = de.compute_portfolio(conn, prices(btc=50_000, gold=1, idr=1))["BTC"]
    assert btc["qty"] == 0 and btc["cost_basis"] == 0
    assert btc["realised_pnl_usd"] == pytest.approx(0.1 * (50_000 - 43_550))
    assert btc["unrealised_pnl_usd"] == 0


def test_cannot_sell_more_than_held(conn):
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 1))
    with pytest.raises(ValueError, match="only 1 is held"):
        de.add_transaction(conn, "BTC", "SELL", 1.5, 50_000, ts=date(2025, 2, 1))
    with pytest.raises(ValueError):  # a sale dated before the purchase
        de.add_transaction(conn, "BTC", "SELL", 0.5, 50_000, ts=date(2024, 12, 1))
    assert len(de.get_transactions(conn)) == 1


def test_back_dated_sale_cannot_break_a_later_sale(conn):
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "SELL", 0.8, 50_000, ts=date(2025, 3, 1))
    with pytest.raises(ValueError):
        de.add_transaction(conn, "BTC", "SELL", 0.5, 45_000, ts=date(2025, 2, 1))


def test_same_day_trades_apply_in_the_order_entered(conn):
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "SELL", 1.0, 41_000, ts=date(2025, 1, 1))
    btc = de.compute_portfolio(conn, prices(btc=50_000, gold=1, idr=1))["BTC"]
    assert btc["qty"] == 0
    assert btc["realised_pnl_usd"] == pytest.approx(1_000)


def test_missing_prices_and_rate_are_none_not_guesses(conn):
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "SELL", 0.5, 60_000, ts=date(2025, 2, 1))
    p = de.compute_portfolio(conn, prices(btc=None, gold=70, idr=None))
    assert p["BTC"]["current_value_usd"] is None
    assert p["BTC"]["pnl_usd"] is None
    assert p["BTC"]["realised_pnl_usd"] == pytest.approx(10_000)  # known without a price
    assert p["GOLD"]["current_value_usd"] == 0  # nothing held
    assert p["GOLD"]["current_value_idr"] is None  # no rate, so no rupiah figure
    assert p["TOTAL"]["pnl_usd"] is None


def test_invalid_input_is_rejected(conn):
    good = dict(asset="BTC", tx_type="BUY", amount=1.0, price_usd=100.0, fee_usd=0.0)
    for bad in [dict(asset="ETH"), dict(tx_type="HOLD"), dict(amount=0), dict(price_usd=-1), dict(fee_usd=-5)]:
        with pytest.raises(ValueError):
            de.add_transaction(conn, **(good | bad))
    assert de.get_transactions(conn).empty


def test_daily_pnl_series_matches_the_portfolio(conn):
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, fee_usd=100, ts=date(2025, 1, 2))
    de.add_transaction(conn, "BTC", "SELL", 0.5, 60_000, fee_usd=50, ts=date(2025, 1, 4))
    days = pd.date_range("2025-01-01", "2025-01-05", freq="D")
    hist = pd.DataFrame({"Close": [39_000, 40_000, 50_000, 60_000, 55_000]}, index=days)
    series = de.daily_pnl_series(conn, "BTC", hist)
    assert list(series.index) == list(days[1:])  # starts on the first trade
    assert series.loc["2025-01-02"] == pytest.approx(-100)  # the buy fee
    assert series.loc["2025-01-03"] == pytest.approx(50_000 - 40_100)
    final = de.compute_portfolio(conn, prices(btc=55_000, gold=1, idr=1))["BTC"]["pnl_usd"]
    assert series.iloc[-1] == pytest.approx(final)


def test_transactions_table_books_profit_on_sells_only(conn):
    de.add_transaction(conn, "GOLD", "BUY", 10, 60, ts=date(2025, 1, 1))
    de.add_transaction(conn, "BTC", "BUY", 1.0, 40_000, ts=date(2025, 1, 2))
    de.add_transaction(conn, "BTC", "SELL", 0.25, 48_000, ts=date(2025, 1, 3))
    table = de.transactions_with_pnl(conn)
    assert list(table["tx_type"]) == ["BUY", "BUY", "SELL"]
    assert table["realised_pnl"].iloc[:2].isna().all()
    assert table["realised_pnl"].iloc[2] == pytest.approx(0.25 * 8_000)


def test_demo_portfolio(tmp_path):
    conn = de.init_db(tmp_path / "demo.db")
    btc = de.compute_portfolio(conn, prices(btc=50_000, gold=100, idr=16_000))["BTC"]
    assert btc["qty"] == pytest.approx(0.10)
    assert btc["avg_cost"] == pytest.approx(43_550)
    conn.close()


def test_demo_trades_are_dated_in_the_past(tmp_path):
    # Seeded with today's date, a fresh demo's profit chart was a single point.
    conn = de.init_db(tmp_path / "demo.db")
    dates = pd.to_datetime(de.get_transactions(conn)["ts"])
    assert len(dates) == 4 and (dates < "2025-01-01").all()
    conn.close()
