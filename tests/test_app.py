"""Runs the whole Streamlit app on fake prices: no network, no errors, and no guessed numbers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import data_engine
import stoic_search

ROOT = Path(__file__).resolve().parent.parent
PRICES = {"BTC_USD": 80_000.0, "GOLD_USD_per_gram": 130.0, "IDR_per_USD": 16_500.0}


def fake_history(ticker_key, period="1y"):
    days = {"5d": 5, "3mo": 92, "6mo": 183, "1y": 365, "2y": 730}[period]
    idx = pd.date_range(end="2026-09-29", periods=days, freq="D")
    if ticker_key == "GOLD":
        idx = idx[idx.dayofweek < 5]
    base = {"BTC": 80_000.0, "GOLD": 130.0}[ticker_key]
    close = base * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx))))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 0}, index=idx
    )


@pytest.fixture
def run_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # portfolio.db and .chromadb stay in the temp folder
    monkeypatch.setattr(data_engine, "DB_PATH", tmp_path / "portfolio.db")
    monkeypatch.setattr(data_engine, "fetch_historical", fake_history)
    monkeypatch.setattr(stoic_search.StoicSearch, "_init_chroma", lambda self, persist_dir: None)
    st.cache_data.clear()
    st.cache_resource.clear()

    def run(prices):
        monkeypatch.setattr(data_engine, "fetch_all_prices", lambda: dict(prices))
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    yield run
    st.cache_data.clear()
    st.cache_resource.clear()


def test_app_runs_and_shows_each_forecast_with_its_track_record(run_app):
    at = run_app(PRICES)
    labels = [m.label for m in at.metric]
    assert "30-day forecast" in labels
    records = [w.value for w in at.warning if "How far to trust this forecast" in w.value]
    assert len(records) == 2  # Bitcoin and gold
    assert any("Meditations" in m.value or "Letter" in m.value or "Enchiridion" in m.value
               or "Discourses" in m.value or "Fragment" in m.value for m in at.markdown)


def test_missing_exchange_rate_is_flagged_not_guessed(run_app):
    at = run_app({**PRICES, "IDR_per_USD": None})
    assert any("USD/IDR rate" in w.value for w in at.warning)
    usd_idr = next(m for m in at.metric if m.label == "💵 USD / IDR")
    assert usd_idr.value == "—"
    idr_values = [m.value for m in at.metric if m.label == "Value (IDR)"]
    assert idr_values and all(v == "—" for v in idr_values)


def test_the_form_refuses_to_sell_more_than_is_held(run_app):
    at = run_app(PRICES)
    form = at.sidebar
    form.selectbox[0].set_value("BTC")
    form.selectbox[1].set_value("SELL")
    form.number_input[0].set_value(0.2)  # the demo portfolio holds 0.1 BTC
    form.number_input[1].set_value(90_000.0)
    form.button[0].click().run()
    assert not at.exception
    assert any("only 0.1 is held" in e.value for e in at.sidebar.error)
    assert len(data_engine.get_transactions(data_engine.init_db(seed_demo=False))) == 4


def test_demo_mode_keeps_each_visitors_trades_private(run_app, monkeypatch, tmp_path):
    monkeypatch.setenv("STOIC_DEMO", "1")
    total = "📊 Total P&L (USD)"
    first = run_app(PRICES)
    assert any(i.value.startswith("Demo:") for i in first.info)
    baseline = next(m.value for m in first.metric if m.label == total)

    form = first.sidebar
    form.selectbox[0].set_value("BTC")
    form.selectbox[1].set_value("BUY")
    form.number_input[0].set_value(0.5)
    form.number_input[1].set_value(60_000.0)  # below the $80,000 price, so an instant gain
    form.button[0].click().run()
    assert not first.exception
    assert next(m.value for m in first.metric if m.label == total) != baseline

    second = run_app(PRICES)  # another visitor still sees only the demo purchases
    assert next(m.value for m in second.metric if m.label == total) == baseline
    assert not (tmp_path / "portfolio.db").exists()


def test_dollar_amounts_are_never_read_as_maths(run_app):
    # Streamlit renders text between two unescaped $ signs as LaTeX, which garbled
    # "Realised: +$0.00 · Unrealised: +$4,029.97" into italic maths on the live demo.
    import re

    at = run_app(PRICES)
    texts = [e.value for e in [*at.markdown, *at.caption, *at.success, *at.info, *at.warning]]
    assert texts
    assert [t for t in texts if len(re.findall(r"(?<!\\)\$", t)) >= 2] == []
