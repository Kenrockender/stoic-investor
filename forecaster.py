"""
forecaster.py
Price forecasts with Prophet, and the backtest track record that says how far
to trust them.

The app shows one model: Prophet with PROPHET_SETTINGS below. backtest.py
scores this exact function against a "no change" guess (the price stays where
it is) and writes results/backtest.json, which the app reads to show each
forecast's track record next to it.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    Prophet = None  # type: ignore
    PROPHET_AVAILABLE = False
    logger.warning("Forecaster: prophet is not installed, so forecasts are turned off")

# cmdstanpy prints two INFO lines for every fit. Giving its logger a handler before
# its first use stops it from attaching its own console handler and resetting the
# level; its warnings still reach the app's log through the root logger.
_cmdstanpy_log = logging.getLogger("cmdstanpy")
_cmdstanpy_log.addHandler(logging.NullHandler())
_cmdstanpy_log.setLevel(logging.WARNING)

# The settings the app has always used. backtest.py imports this module, so the
# backtest always measures the model the app shows.
PROPHET_SETTINGS = dict(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.15,
    interval_width=0.80,
)
MIN_HISTORY_ROWS = 30
BACKTEST_PATH = Path(__file__).resolve().parent / "results" / "backtest.json"


def to_prophet_frame(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV frame (DatetimeIndex + 'Close') -> Prophet's ds/y frame, one row per day."""
    df = hist_df["Close"].dropna().reset_index()
    df.columns = ["ds", "y"]
    ds = pd.to_datetime(df["ds"])
    if ds.dt.tz is not None:
        ds = ds.dt.tz_localize(None)
    df["ds"] = ds.dt.normalize()
    df = df.drop_duplicates("ds", keep="last")
    return df.sort_values("ds").reset_index(drop=True)


def trades_on_weekends(ds: pd.Series) -> bool:
    """True for assets priced every day (Bitcoin), False for weekday-only ones (gold futures)."""
    return bool((pd.to_datetime(ds).dt.dayofweek >= 5).mean() > 0.05)


def prophet_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Fit Prophet on a ds/y frame and forecast up to `periods` calendar days after its last date.

    Returns the future rows only (ds, yhat, yhat_lower, yhat_upper). Weekday-only assets
    get no weekend rows, because there is no price to forecast on those days. Raises if
    Prophet is missing or fails; the caller decides what to show.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("prophet is not installed")
    last = pd.Timestamp(df["ds"].max())
    future = pd.DataFrame(
        {"ds": pd.date_range(last + pd.Timedelta(days=1), periods=periods, freq="D")}
    )
    if not trades_on_weekends(df["ds"]):
        future = future[future["ds"].dt.dayofweek < 5]
    if future.empty:
        raise ValueError(f"no trading days in the next {periods} days")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(**PROPHET_SETTINGS)
        model.fit(df[["ds", "y"]])
        fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


def forecast(hist_df: Optional[pd.DataFrame], periods: int = 30) -> Optional[pd.DataFrame]:
    """The app's forecast: future rows, or None when there is too little data or Prophet fails."""
    if hist_df is None or hist_df.empty or "Close" not in hist_df.columns:
        return None
    df = to_prophet_frame(hist_df)
    if len(df) < MIN_HISTORY_ROWS:
        logger.warning("Forecaster: only %d days of history, need %d", len(df), MIN_HISTORY_ROWS)
        return None
    try:
        return prophet_forecast(df, periods)
    except Exception as exc:
        logger.error("Forecaster: Prophet failed: %s", exc)
        return None


def forecast_summary(fcst_df: Optional[pd.DataFrame], current_price: Optional[float]) -> dict:
    """Headline numbers for the last day of the forecast."""
    if fcst_df is None or fcst_df.empty:
        return {}
    last = fcst_df.iloc[-1]
    change_pct = None
    if current_price:
        change_pct = (float(last["yhat"]) - current_price) / current_price * 100
    return {
        "target_date": pd.Timestamp(last["ds"]),
        "target_price": float(last["yhat"]),
        "target_lower": float(last["yhat_lower"]),
        "target_upper": float(last["yhat_upper"]),
        "change_pct": change_pct,
    }


def load_backtest(path: Path = BACKTEST_PATH) -> Optional[dict]:
    """results/backtest.json as a dict, or None when the backtest has not been run."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def track_record(
    backtest: Optional[dict], asset: str, horizon_days: int, history_period: str
) -> Optional[dict]:
    """
    How this forecast did in the backtest for this asset and horizon. None when the
    backtest did not cover this setup: it measures the app's default history only.
    """
    if not backtest or backtest.get("method", {}).get("history_period") != history_period:
        return None
    asset_result = backtest.get("assets", {}).get(asset)
    if not asset_result:
        return None
    stats = asset_result.get("by_horizon", {}).get(str(int(horizon_days)))
    if not stats:
        return None
    return {
        **stats,
        "first_origin": asset_result["first_origin"],
        "last_origin": asset_result["last_origin"],
    }
