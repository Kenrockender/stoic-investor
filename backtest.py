"""
backtest.py: how far can the app's price forecast be trusted?

For each asset, every 14 days from the first date with a full year of history:
  1. take the 365 days of daily closes up to that day (the app's default "1y" history),
  2. forecast 90 days ahead with forecaster.prophet_forecast, the function the app uses,
  3. score every horizon h = 1..90 on the last trading day on or before day + h, against
     the simplest guess, "no change": the price stays where it was on the forecast day.

Writes results/backtest.json (summary per asset and horizon; the app reads it) and
results/backtest_errors.csv (per-forecast % errors at 7, 30 and 90 days; no prices).

Run (a few minutes on 8 cores):
    python backtest.py                      # downloads prices with yfinance into data/prices.csv
    python backtest.py --prices other.csv   # or a saved file with date, btc, gold_futures columns
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import forecaster

ROOT = Path(__file__).resolve().parent
DEFAULT_PRICES = ROOT / "data" / "prices.csv"
RESULTS_DIR = ROOT / "results"

HISTORY_DAYS = 365  # the app's default history setting, "1y"
HISTORY_PERIOD = "1y"
STEP_DAYS = 14
MAX_HORIZON = 90  # the app's longest forecast horizon
CSV_HORIZONS = (7, 30, 90)
COLUMNS = {"BTC": "btc", "GOLD": "gold_futures"}  # asset -> prices.csv column
TICKERS = {"BTC": "BTC-USD", "GOLD": "GC=F"}  # the tickers the app uses


def download_prices(path: Path) -> None:
    """Save the full daily close history of both tickers, the way the app downloads them."""
    import yfinance as yf

    closes = {}
    for asset, ticker in TICKERS.items():
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            sys.exit(f"Yahoo Finance returned no data for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes[COLUMNS[asset]] = df["Close"]
    out = pd.DataFrame(closes)
    index = pd.to_datetime(out.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, float_format="%.4f")


def load_prices(path: Path) -> dict:
    """{asset: Series of daily closes on that asset's own trading days}."""
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    series = {}
    for asset, column in COLUMNS.items():
        s = raw[column].dropna()
        series[asset] = s[s > 0]
    return series


def forecast_days(series: dict, step_days: int = STEP_DAYS) -> list:
    """One shared calendar: every asset has a full history before each day and 90 days after."""
    start = max(s.index.min() for s in series.values()) + pd.Timedelta(days=HISTORY_DAYS)
    end = min(s.index.max() for s in series.values()) - pd.Timedelta(days=MAX_HORIZON)
    return list(pd.date_range(start, end, freq=f"{step_days}D"))


def build_jobs(series: dict, days: list) -> list:
    """(asset, origin, training closes, [(horizon, target date, actual close)]) per forecast."""
    jobs = []
    for asset, s in series.items():
        index = s.index
        for day in days:
            origin = index[index.searchsorted(day, side="right") - 1]  # last close on or before
            train = s[(index > origin - pd.Timedelta(days=HISTORY_DAYS)) & (index <= origin)]
            targets = []
            for h in range(1, MAX_HORIZON + 1):
                j = index.searchsorted(origin + pd.Timedelta(days=h), side="right") - 1
                if index[j] > origin:  # a gold "1 day ahead" from a Friday has no new close yet
                    targets.append((h, index[j], float(s.iloc[j])))
            jobs.append((asset, origin, train, targets))
    return jobs


def score(asset, origin, h, start_price, actual, yhat, lower, upper) -> dict:
    """Percentage errors of the forecast and of 'no change' for one horizon."""
    move = actual - start_price
    return {
        "asset": asset,
        "origin": pd.Timestamp(origin).date().isoformat(),
        "horizon": h,
        "ape_model": abs(yhat - actual) / actual * 100,
        "ape_naive": abs(start_price - actual) / actual * 100,
        "direction_hit": None if move == 0 else bool(np.sign(yhat - start_price) == np.sign(move)),
        "inside_band": bool(lower <= actual <= upper),
    }


def run_job(job, model=forecaster.prophet_forecast) -> list:
    asset, origin, train, targets = job
    np.random.seed(0)  # Prophet samples its uncertainty band; a fixed seed makes runs repeatable
    fc = model(pd.DataFrame({"ds": train.index, "y": train.values}), MAX_HORIZON).set_index("ds")
    start_price = float(train.iloc[-1])
    rows = []
    for h, target_date, actual in targets:
        f = fc.loc[target_date]
        rows.append(
            score(asset, origin, h, start_price, actual, f["yhat"], f["yhat_lower"], f["yhat_upper"])
        )
    return rows


def summarise(rows: pd.DataFrame) -> dict:
    out = {}
    for asset, g in rows.groupby("asset"):
        by_horizon = {}
        for h, gh in g.groupby("horizon"):
            hits = gh["direction_hit"].dropna().astype(bool)
            by_horizon[str(int(h))] = {
                "n": int(len(gh)),
                "mape_model": round(float(gh["ape_model"].mean()), 2),
                "mape_naive": round(float(gh["ape_naive"].mean()), 2),
                "median_ape_model": round(float(gh["ape_model"].median()), 2),
                "median_ape_naive": round(float(gh["ape_naive"].median()), 2),
                "win_rate": round(float((gh["ape_model"] < gh["ape_naive"]).mean() * 100), 1),
                "direction_hit_rate": round(float(hits.mean() * 100), 1) if len(hits) else None,
                "band_coverage": round(float(gh["inside_band"].mean() * 100), 1),
            }
        out[asset] = {
            "ticker": TICKERS[asset],
            "n_forecasts": int(g["origin"].nunique()),
            "first_origin": str(g["origin"].min()),
            "last_origin": str(g["origin"].max()),
            "by_horizon": by_horizon,
        }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Backtest the app's Prophet forecast.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--step-days", type=int, default=STEP_DAYS)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--limit", type=int, help="only the first N forecast days (a quick check)")
    args = parser.parse_args(argv)

    if not args.prices.exists():
        if args.prices != DEFAULT_PRICES:
            sys.exit(f"{args.prices} not found")
        print("Downloading prices from Yahoo Finance...", flush=True)
        download_prices(args.prices)
    series = load_prices(args.prices)
    days = forecast_days(series, args.step_days)[: args.limit]
    jobs = build_jobs(series, days)

    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, job_rows in enumerate(pool.map(run_job, jobs, chunksize=2), 1):
            rows.extend(job_rows)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} forecasts, {time.time() - started:.0f}s", flush=True)
    errors = pd.DataFrame(rows)

    import prophet

    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "data": {
            "source": "Yahoo Finance daily closes",
            "tickers": TICKERS,
            "first_date": {a: str(s.index.min().date()) for a, s in series.items()},
            "last_date": {a: str(s.index.max().date()) for a, s in series.items()},
            "prices_sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        },
        "method": {
            "model": f"Prophet {prophet.__version__} with the app's settings",
            "settings": forecaster.PROPHET_SETTINGS,
            "baseline": "no change: the close on the forecast day",
            "history_period": HISTORY_PERIOD,
            "history_days": HISTORY_DAYS,
            "step_days": args.step_days,
            "max_horizon_days": MAX_HORIZON,
            "scored_on": "the last close on or before forecast day + horizon",
            "metrics": {
                "mape": "mean absolute error, % of the actual price",
                "win_rate": "% of forecasts closer to the actual price than 'no change'",
                "direction_hit_rate": "% of forecasts that got the direction of the move right",
                "band_coverage": "% of actual prices inside the forecast's 80% band",
            },
        },
        "assets": summarise(errors),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "backtest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    subset = errors[errors["horizon"].isin(CSV_HORIZONS)].copy()
    subset[["ape_model", "ape_naive"]] = subset[["ape_model", "ape_naive"]].round(3)
    subset.to_csv(args.out_dir / "backtest_errors.csv", index=False)

    print(f"\nWrote {args.out_dir / 'backtest.json'}")
    for asset, res in result["assets"].items():
        print(f"\n{asset}: {res['n_forecasts']} forecasts, {res['first_origin']} to {res['last_origin']}")
        print("  days  prophet_mape  nochange_mape  closer%  direction%  band80%")
        for h in CSV_HORIZONS:
            r = res["by_horizon"][str(h)]
            print(
                f"  {h:>4}  {r['mape_model']:>12}  {r['mape_naive']:>13}  {r['win_rate']:>7}"
                f"  {r['direction_hit_rate']:>10}  {r['band_coverage']:>7}"
            )


if __name__ == "__main__":
    main()
