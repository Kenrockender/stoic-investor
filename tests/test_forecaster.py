"""The forecast the app shows, and the backtest that scores it."""

import numpy as np
import pandas as pd
import pytest

import backtest
import forecaster

needs_prophet = pytest.mark.skipif(not forecaster.PROPHET_AVAILABLE, reason="prophet not installed")


def daily_closes(days=400, start="2024-01-01", weekdays_only=False, seed=1) -> pd.DataFrame:
    idx = pd.date_range(start, periods=days, freq="D")
    if weekdays_only:
        idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0}, index=idx)


@needs_prophet
def test_forecast_returns_only_future_days():
    hist = daily_closes()
    fc = forecaster.forecast(hist, periods=30)
    assert len(fc) == 30
    assert fc["ds"].min() == hist.index.max() + pd.Timedelta(days=1)
    assert (fc["yhat_lower"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["yhat_upper"]).all()


@needs_prophet
def test_weekday_only_assets_get_no_weekend_forecasts():
    hist = daily_closes(weekdays_only=True)
    fc = forecaster.forecast(hist, periods=30)
    assert (fc["ds"].dt.dayofweek < 5).all()
    assert fc["ds"].max() <= hist.index.max() + pd.Timedelta(days=30)


def test_too_little_history_gives_no_forecast():
    assert forecaster.forecast(daily_closes(days=10), periods=30) is None
    assert forecaster.forecast(pd.DataFrame(), periods=30) is None


def test_summary_uses_the_last_forecast_day():
    fc = pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=3),
        "yhat": [10.0, 11.0, 12.0], "yhat_lower": [9.0, 9.5, 10.0], "yhat_upper": [11.0, 12.5, 14.0],
    })
    s = forecaster.forecast_summary(fc, current_price=10.0)
    assert (s["target_price"], s["target_lower"], s["target_upper"]) == (12.0, 10.0, 14.0)
    assert s["change_pct"] == pytest.approx(20.0)
    assert forecaster.forecast_summary(fc, current_price=None)["change_pct"] is None
    assert forecaster.forecast_summary(None, current_price=10.0) == {}


def test_track_record_only_for_the_backtested_setup():
    bt = {"method": {"history_period": "1y"},
          "assets": {"BTC": {"first_origin": "2015-09-17", "last_origin": "2026-06-25",
                             "by_horizon": {"30": {"n": 282, "mape_model": 24.2, "mape_naive": 15.0}}}}}
    record = forecaster.track_record(bt, "BTC", 30, "1y")
    assert record["mape_model"] == 24.2 and record["first_origin"] == "2015-09-17"
    assert forecaster.track_record(bt, "BTC", 30, "2y") is None
    assert forecaster.track_record(bt, "GOLD", 30, "1y") is None
    assert forecaster.track_record(None, "BTC", 30, "1y") is None


def test_published_backtest_covers_every_horizon_the_app_offers():
    bt = forecaster.load_backtest()
    assert bt is not None, "results/backtest.json is missing: run python backtest.py"
    assert bt["method"]["history_period"] == "1y"
    for asset in ("BTC", "GOLD"):
        horizons = bt["assets"][asset]["by_horizon"]
        assert all(str(h) in horizons for h in range(7, 91)), asset


def test_backtest_scores_on_the_last_close_on_or_before_each_target():
    # Gold has no weekend closes: a forecast made on a Saturday starts from Friday's close,
    # and a target that lands on a weekend is scored on the Friday before it.
    idx = pd.bdate_range("2024-01-01", "2025-06-30")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    (asset, origin, train, targets), = backtest.build_jobs({"GOLD": s}, [pd.Timestamp("2025-01-04")])
    assert origin == pd.Timestamp("2025-01-03")
    assert train.index.max() == origin
    assert train.index.min() > origin - pd.Timedelta(days=backtest.HISTORY_DAYS)
    dates = {h: d for h, d, _ in targets}
    assert 1 not in dates and 2 not in dates  # Saturday and Sunday bring no new close
    assert dates[3] == pd.Timestamp("2025-01-06")
    assert dates[8] == pd.Timestamp("2025-01-10")


def test_scores_and_summary_for_a_model_with_known_errors():
    # The price never moves, so "no change" is perfect; the model always says +10%.
    s = pd.Series(100.0, index=pd.date_range("2024-01-01", "2025-12-31", freq="D"))

    def plus_ten(df, periods):
        ds = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=periods)
        y = float(df["y"].iloc[-1]) * 1.10
        return pd.DataFrame({"ds": ds, "yhat": y, "yhat_lower": y * 0.95, "yhat_upper": y * 1.05})

    jobs = backtest.build_jobs({"BTC": s}, [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")])
    rows = [row for job in jobs for row in backtest.run_job(job, model=plus_ten)]
    h30 = backtest.summarise(pd.DataFrame(rows))["BTC"]["by_horizon"]["30"]
    assert h30["n"] == 2
    assert h30["mape_model"] == pytest.approx(10.0)
    assert h30["mape_naive"] == 0
    assert h30["win_rate"] == 0
    assert h30["band_coverage"] == 0  # 100 is outside the 104.5 to 115.5 band
    assert h30["direction_hit_rate"] is None  # no move to call
