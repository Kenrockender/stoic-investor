"""
forecaster.py
Time-series price forecasting using Facebook Prophet (or NeuralProphet if available).
Returns a forecast DataFrame and a confidence-band plot.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try NeuralProphet first, fall back to Prophet, then to a naive trend model
_BACKEND = "none"
try:
    from neuralprophet import NeuralProphet   # type: ignore
    _BACKEND = "neuralprophet"
    logger.info("Forecaster: using NeuralProphet")
except ImportError:
    try:
        from prophet import Prophet            # type: ignore
        _BACKEND = "prophet"
        logger.info("Forecaster: using Prophet")
    except ImportError:
        logger.warning("Forecaster: neither Prophet nor NeuralProphet found — using naive trend")


# ---------------------------------------------------------------------------
# Core forecast function
# ---------------------------------------------------------------------------

def forecast(
    hist_df: pd.DataFrame,
    periods: int = 30,
    asset_name: str = "asset",
) -> Optional[pd.DataFrame]:
    """
    Given a historical OHLCV DataFrame (indexed by date, with a 'Close' column),
    return a forecast DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper
    Returns None on failure.
    """
    if hist_df is None or hist_df.empty:
        return None
    if "Close" not in hist_df.columns:
        return None

    # Build prophet-style input
    df = hist_df["Close"].dropna().reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < 30:
        logger.warning("Forecaster: not enough data (%d rows)", len(df))
        return _naive_forecast(df, periods)

    if _BACKEND == "neuralprophet":
        return _neuralprophet_forecast(df, periods)
    elif _BACKEND == "prophet":
        return _prophet_forecast(df, periods)
    else:
        return _naive_forecast(df, periods)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _prophet_forecast(df: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
    try:
        from prophet import Prophet
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.15,
                interval_width=0.80,
            )
            model.fit(df)
            future  = model.make_future_dataframe(periods=periods)
            forecast_df = model.predict(future)
            result = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            result["ds"] = pd.to_datetime(result["ds"])
            return result
    except Exception as exc:
        logger.error("Prophet forecast failed: %s", exc)
        return _naive_forecast(df, periods)


def _neuralprophet_forecast(df: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
    try:
        from neuralprophet import NeuralProphet
        model = NeuralProphet(
            n_forecasts=periods,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            quantiles=[0.1, 0.9],
        )
        split = int(len(df) * 0.85)
        train_df = df.iloc[:split]
        model.fit(train_df, freq="D", progress="none")
        future = model.make_future_dataframe(df, periods=periods)
        fcst   = model.predict(future)

        # NeuralProphet column naming
        yhat_col = "yhat1"
        lo_col   = next((c for c in fcst.columns if "10" in c), None)
        hi_col   = next((c for c in fcst.columns if "90" in c), None)

        result = pd.DataFrame({
            "ds":         pd.to_datetime(fcst["ds"]),
            "yhat":       fcst[yhat_col],
            "yhat_lower": fcst[lo_col] if lo_col else fcst[yhat_col] * 0.92,
            "yhat_upper": fcst[hi_col] if hi_col else fcst[yhat_col] * 1.08,
        })
        return result
    except Exception as exc:
        logger.error("NeuralProphet forecast failed: %s", exc)
        return _naive_forecast(df, periods)


def _naive_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Linear + trend-noise naive model as a last resort.
    Fits a simple linear regression on log-price and extrapolates.
    """
    n = len(df)
    log_y = np.log(df["y"].clip(lower=1e-6).values)
    x     = np.arange(n)

    # Fit linear trend on last 90 days (or all data)
    window = min(90, n)
    x_w = x[-window:]
    y_w = log_y[-window:]
    coeffs = np.polyfit(x_w - x_w.mean(), y_w, 1)

    last_date = df["ds"].iloc[-1]
    future_x  = np.arange(1, periods + 1)
    log_pred  = coeffs[0] * (future_x + x[-1] - x_w.mean()) + coeffs[1]
    pred      = np.exp(log_pred)

    # Historical residual std for confidence bands
    hist_pred = np.exp(np.polyval(coeffs, x - x_w.mean()))
    resid_std = np.std(log_y - np.polyval(coeffs, x - x_w.mean()))
    band_pct  = np.exp(1.28 * resid_std)   # ~80% interval

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    historical = pd.DataFrame({
        "ds":         df["ds"],
        "yhat":       np.exp(np.polyval(coeffs, x - x_w.mean())),
        "yhat_lower": np.exp(np.polyval(coeffs, x - x_w.mean())) / band_pct,
        "yhat_upper": np.exp(np.polyval(coeffs, x - x_w.mean())) * band_pct,
    })
    future_rows = pd.DataFrame({
        "ds":         future_dates,
        "yhat":       pred,
        "yhat_lower": pred / band_pct,
        "yhat_upper": pred * band_pct,
    })
    return pd.concat([historical, future_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Forecast summary helper
# ---------------------------------------------------------------------------

def forecast_summary(fcst_df: pd.DataFrame, current_price: float) -> dict:
    """Return a dict with key forecast stats for display."""
    if fcst_df is None or fcst_df.empty:
        return {}
    last_fcst  = fcst_df.iloc[-1]
    mid_fcst   = fcst_df.iloc[len(fcst_df) // 2]
    change_pct = (last_fcst["yhat"] - current_price) / current_price * 100 if current_price else 0
    return {
        "target_price":    last_fcst["yhat"],
        "target_lower":    last_fcst["yhat_lower"],
        "target_upper":    last_fcst["yhat_upper"],
        "mid_price":       mid_fcst["yhat"],
        "change_pct":      change_pct,
        "horizon_days":    (fcst_df["ds"].iloc[-1] - fcst_df["ds"].iloc[0]).days,
        "bullish":         change_pct > 0,
    }
