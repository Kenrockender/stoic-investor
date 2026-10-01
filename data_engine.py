"""
data_engine.py
Handles:
  - Live & historical price fetching via yfinance (BTC-USD, GC=F gold, IDR=X)
  - Transaction history stored in a local SQLite database
  - Portfolio P&L calculations
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path("portfolio.db")

# Ticker symbols
TICKERS = {
    "BTC":  "BTC-USD",   # Bitcoin in USD
    "GOLD": "GC=F",      # Gold futures (USD per troy oz)
    "IDR":  "IDR=X",     # USD/IDR exchange rate
}

GOLD_GRAMS_PER_OZ = 31.1035  # 1 troy oz = 31.1035 grams


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset       TEXT    NOT NULL,          -- 'BTC' | 'GOLD'
            tx_type     TEXT    NOT NULL,          -- 'BUY' | 'SELL'
            amount      REAL    NOT NULL,          -- qty (BTC units | grams of gold)
            price_usd   REAL    NOT NULL,          -- price per unit in USD at tx time
            price_idr   REAL,                      -- price per unit in IDR (optional)
            fee_usd     REAL    DEFAULT 0,
            note        TEXT,
            ts          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker      TEXT NOT NULL,
            ts          TEXT NOT NULL,
            price_usd   REAL NOT NULL,
            PRIMARY KEY (ticker, ts)
        )
    """)
    conn.commit()
    # Seed demo transactions if table is empty
    _seed_demo(conn)
    return conn


def _seed_demo(conn: sqlite3.Connection):
    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    if count == 0:
        demo = [
            # asset, tx_type, amount, price_usd, note
            ("GOLD", "BUY",  125.0,  63.5,   "Initial 125g gold purchase"),   # 125g ≈ 4.02 oz @ $63.5/g
            ("BTC",  "BUY",  0.05,   42000.0, "DCA entry Jan"),
            ("BTC",  "BUY",  0.03,   38500.0, "DCA entry Feb (dip)"),
            ("BTC",  "BUY",  0.02,   55000.0, "DCA entry Mar"),
        ]
        conn.executemany(
            """INSERT INTO transactions (asset, tx_type, amount, price_usd, note)
               VALUES (?, ?, ?, ?, ?)""",
            demo,
        )
        conn.commit()
        logger.info("DataEngine: seeded %d demo transactions", len(demo))


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_price(ticker_key: str) -> Optional[float]:
    """Return the latest price for a ticker key (BTC | GOLD | IDR). USD-denominated."""
    symbol = TICKERS.get(ticker_key)
    if not symbol:
        raise ValueError(f"Unknown ticker key: {ticker_key}")
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1d", interval="1m")
        if hist.empty:
            hist = t.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.error("fetch_price(%s): %s", ticker_key, exc)
        return None


def fetch_historical(ticker_key: str, period: str = "1y") -> pd.DataFrame:
    """
    Return OHLCV dataframe.
    For GOLD: price is in USD per troy oz → converted to USD per gram.
    """
    symbol = TICKERS[ticker_key]
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if ticker_key == "GOLD":
            # Convert troy oz → gram
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col] / GOLD_GRAMS_PER_OZ
        return df
    except Exception as exc:
        logger.error("fetch_historical(%s): %s", ticker_key, exc)
        return pd.DataFrame()


def fetch_all_prices() -> dict[str, Optional[float]]:
    """Return {BTC_USD, GOLD_USD_per_gram, IDR_per_USD}."""
    btc  = fetch_price("BTC")
    gold_oz = fetch_price("GOLD")
    idr  = fetch_price("IDR")
    return {
        "BTC_USD":          btc,
        "GOLD_USD_per_gram": gold_oz / GOLD_GRAMS_PER_OZ if gold_oz else None,
        "IDR_per_USD":      idr,
    }


def to_idr(usd_value: Optional[float], idr_rate: Optional[float]) -> Optional[float]:
    if usd_value is None or idr_rate is None:
        return None
    return usd_value * idr_rate


# ---------------------------------------------------------------------------
# Transaction helpers
# ---------------------------------------------------------------------------

def add_transaction(
    conn: sqlite3.Connection,
    asset: str,
    tx_type: str,
    amount: float,
    price_usd: float,
    price_idr: Optional[float] = None,
    fee_usd: float = 0.0,
    note: str = "",
    ts: Optional[str] = None,
):
    ts = ts or datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn.execute(
        """INSERT INTO transactions
           (asset, tx_type, amount, price_usd, price_idr, fee_usd, note, ts)
           VALUES (?,?,?,?,?,?,?,?)""",
        (asset.upper(), tx_type.upper(), amount, price_usd, price_idr, fee_usd, note, ts),
    )
    conn.commit()


def get_transactions(conn: sqlite3.Connection, asset: Optional[str] = None) -> pd.DataFrame:
    query = "SELECT * FROM transactions"
    params: tuple = ()
    if asset:
        query += " WHERE asset = ?"
        params = (asset.upper(),)
    query += " ORDER BY ts"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


# ---------------------------------------------------------------------------
# Portfolio calculations
# ---------------------------------------------------------------------------

def compute_portfolio(conn: sqlite3.Connection, prices: dict) -> dict:
    """
    Returns a dict with holdings, cost basis, current value, P&L for each asset
    plus totals in USD and IDR.
    """
    idr = prices.get("IDR_per_USD") or 15_800  # fallback

    result = {}
    for asset, price_key in [("BTC", "BTC_USD"), ("GOLD", "GOLD_USD_per_gram")]:
        df = get_transactions(conn, asset)
        if df.empty:
            result[asset] = {
                "qty": 0, "avg_cost": 0, "cost_basis": 0,
                "current_price": prices.get(price_key, 0) or 0,
                "current_value_usd": 0, "pnl_usd": 0,
                "current_value_idr": 0, "pnl_idr": 0,
                "pnl_pct": 0,
            }
            continue

        buys  = df[df["tx_type"] == "BUY"]
        sells = df[df["tx_type"] == "SELL"]

        qty_bought = buys["amount"].sum()
        qty_sold   = sells["amount"].sum()
        qty        = qty_bought - qty_sold

        cost_basis = (buys["amount"] * buys["price_usd"]).sum() \
                   - (sells["amount"] * sells["price_usd"]).sum()
        avg_cost   = cost_basis / qty if qty > 0 else 0

        cur_price       = prices.get(price_key) or 0
        current_value   = qty * cur_price
        pnl             = current_value - cost_basis
        pnl_pct         = (pnl / cost_basis * 100) if cost_basis else 0

        result[asset] = {
            "qty":               qty,
            "avg_cost":          avg_cost,
            "cost_basis":        cost_basis,
            "current_price":     cur_price,
            "current_value_usd": current_value,
            "pnl_usd":           pnl,
            "current_value_idr": current_value * idr,
            "pnl_idr":           pnl * idr,
            "pnl_pct":           pnl_pct,
        }

    # Totals
    total_cost  = sum(v["cost_basis"] for v in result.values())
    total_value = sum(v["current_value_usd"] for v in result.values())
    total_pnl   = total_value - total_cost

    result["TOTAL"] = {
        "cost_basis":        total_cost,
        "current_value_usd": total_value,
        "pnl_usd":           total_pnl,
        "current_value_idr": total_value * idr,
        "pnl_idr":           total_pnl  * idr,
        "pnl_pct":           (total_pnl / total_cost * 100) if total_cost else 0,
    }
    return result


def daily_pnl_series(conn: sqlite3.Connection, asset: str, hist_df: pd.DataFrame) -> pd.Series:
    """Approximate daily portfolio value series for one asset."""
    txns = get_transactions(conn, asset)
    if txns.empty or hist_df.empty:
        return pd.Series(dtype=float)

    # Reindex hist_df to business days
    close = hist_df["Close"].copy()
    pnl_series = []
    for date, price in close.items():
        day = pd.Timestamp(date).normalize()
        past = txns[txns["ts"].dt.normalize() <= day]
        if past.empty:
            continue
        buys  = past[past["tx_type"] == "BUY"]
        sells = past[past["tx_type"] == "SELL"]
        qty   = buys["amount"].sum() - sells["amount"].sum()
        cost  = (buys["amount"] * buys["price_usd"]).sum() \
              - (sells["amount"] * sells["price_usd"]).sum()
        pnl_series.append({"date": day, "value": qty * price, "cost": cost})

    if not pnl_series:
        return pd.Series(dtype=float)
    df2 = pd.DataFrame(pnl_series).set_index("date")
    return df2["value"] - df2["cost"]
