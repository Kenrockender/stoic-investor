"""
data_engine.py
Handles:
  - Live & historical prices from Yahoo Finance (BTC-USD, GC=F gold, IDR=X)
  - Transaction history stored in a local SQLite database
  - Portfolio profit and loss, with the average-cost method

Average-cost method, applied to each asset's transactions in date order:
  - a BUY adds its units, and what they cost including the fee, to the position;
  - a SELL takes units out at the position's average cost. What it sold for, minus
    its fee, less what those units cost, is REALISED profit;
  - the units still held, valued at the live price, less what they cost, are
    UNREALISED profit. Total profit is the two added together.

A value that needs a price or exchange rate that failed to load is None, never a
guess, so the app can say it is missing instead of showing a wrong number.
"""

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("STOIC_DB_PATH", "portfolio.db"))

# Ticker symbols
TICKERS = {
    "BTC":  "BTC-USD",   # Bitcoin in USD
    "GOLD": "GC=F",      # COMEX gold futures (USD per troy oz)
    "IDR":  "IDR=X",     # USD/IDR exchange rate
}
ASSETS = ("BTC", "GOLD")
TX_TYPES = ("BUY", "SELL")

GOLD_GRAMS_PER_OZ = 31.1035  # 1 troy oz = 31.1035 grams
QTY_TOLERANCE = 1e-9  # float slack, so selling "everything" never leaves dust or an error


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def init_db(db_path: Optional[Path] = None, seed_demo: bool = True) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
    conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
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
    if seed_demo:
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
    """Return the latest price for a ticker key (BTC | GOLD | IDR), or None if it failed to load."""
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
    """Return {BTC_USD, GOLD_USD_per_gram, IDR_per_USD}; a price that failed to load is None."""
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
# Average-cost position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """One asset's position after its transactions are applied in date order. USD throughout."""
    qty: float = 0.0
    cost_basis: float = 0.0    # what the units still held cost, buy fees included
    realised_pnl: float = 0.0  # profit taken on sells, after their fees
    invested: float = 0.0      # everything ever paid on buys, fees included
    fees: float = 0.0          # every fee paid

    @property
    def avg_cost(self) -> float:
        return self.cost_basis / self.qty if self.qty > QTY_TOLERANCE else 0.0

    def apply(self, tx_type: str, amount: float, price: float, fee: float = 0.0) -> float:
        """Apply one transaction. Returns the realised profit of a SELL, 0 for a BUY."""
        if tx_type == "BUY":
            self.qty += amount
            self.cost_basis += amount * price + fee
            self.invested += amount * price + fee
            self.fees += fee
            return 0.0
        if tx_type == "SELL":
            if amount > self.qty + QTY_TOLERANCE:
                raise ValueError(f"sells {amount:g} when only {self.qty:g} is held")
            cost_of_sold = self.avg_cost * amount
            realised = amount * price - fee - cost_of_sold
            self.qty -= amount
            self.cost_basis -= cost_of_sold
            if self.qty <= QTY_TOLERANCE:  # sold out: clear float dust
                self.qty = 0.0
                self.cost_basis = 0.0
            self.realised_pnl += realised
            self.fees += fee
            return realised
        raise ValueError(f"unknown transaction type {tx_type!r}")


def _fee(value) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def replay(txns: pd.DataFrame) -> tuple[Position, list[float]]:
    """
    Apply transactions (already in date order) to a fresh position.
    Returns the position and each transaction's realised profit (0 for buys).
    Raises ValueError, naming the date, if a sale would sell more than was held.
    """
    pos = Position()
    realised = []
    if txns is None or txns.empty:
        return pos, realised
    for row in txns.itertuples(index=False):
        try:
            realised.append(pos.apply(row.tx_type, float(row.amount), float(row.price_usd), _fee(row.fee_usd)))
        except ValueError as exc:
            raise ValueError(f"{row.asset} {row.tx_type} on {pd.Timestamp(row.ts).date()} {exc}") from None
    return pos, realised


# ---------------------------------------------------------------------------
# Transaction helpers
# ---------------------------------------------------------------------------

def _to_ts(ts: Union[None, str, date, datetime]) -> str:
    if ts is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, date):
        return f"{ts.isoformat()} 00:00:00"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def add_transaction(
    conn: sqlite3.Connection,
    asset: str,
    tx_type: str,
    amount: float,
    price_usd: float,
    price_idr: Optional[float] = None,
    fee_usd: float = 0.0,
    note: str = "",
    ts: Union[None, str, date, datetime] = None,
):
    """
    Record a transaction. `ts` is the trade date (a date, datetime or ISO string; default
    now, UTC). Trades on the same day are applied in the order they were entered.
    Raises ValueError for bad input, or for a sale of more than is held on that date,
    including a back-dated sale that would make a later sale impossible.
    """
    asset, tx_type = asset.upper(), tx_type.upper()
    if asset not in ASSETS:
        raise ValueError(f"asset must be one of {ASSETS}")
    if tx_type not in TX_TYPES:
        raise ValueError(f"type must be one of {TX_TYPES}")
    if not amount or amount <= 0:
        raise ValueError("amount must be more than 0")
    if not price_usd or price_usd <= 0:
        raise ValueError("price must be more than 0")
    if fee_usd is None or fee_usd < 0:
        raise ValueError("fee cannot be negative")
    ts = _to_ts(ts)

    # Replay the ledger with the new trade in place, so no sale ever exceeds holdings.
    existing = get_transactions(conn, asset)
    new_row = pd.DataFrame([{
        "id": (existing["id"].max() + 1) if not existing.empty else 1,
        "asset": asset, "tx_type": tx_type, "amount": float(amount),
        "price_usd": float(price_usd), "fee_usd": float(fee_usd), "ts": pd.Timestamp(ts),
    }])
    trial = pd.concat([existing, new_row], ignore_index=True) if not existing.empty else new_row
    trial = trial.assign(_day=trial["ts"].dt.normalize()).sort_values(["_day", "id"], kind="stable")
    try:
        replay(trial)
    except ValueError as exc:
        raise ValueError(f"Cannot record this trade: {exc}") from None

    conn.execute(
        """INSERT INTO transactions
           (asset, tx_type, amount, price_usd, price_idr, fee_usd, note, ts)
           VALUES (?,?,?,?,?,?,?,?)""",
        (asset, tx_type, amount, price_usd, price_idr, fee_usd, note, ts),
    )
    conn.commit()


def get_transactions(conn: sqlite3.Connection, asset: Optional[str] = None) -> pd.DataFrame:
    """Transactions in the order they are applied: by trade date, then the order entered."""
    query = "SELECT * FROM transactions"
    params: tuple = ()
    if asset:
        query += " WHERE asset = ?"
        params = (asset.upper(),)
    query += " ORDER BY date(ts), id"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    return df


def transactions_with_pnl(conn: sqlite3.Connection) -> pd.DataFrame:
    """All transactions, each with the realised profit it booked (sells only; NaN for buys)."""
    frames = []
    for asset in ASSETS:
        txns = get_transactions(conn, asset)
        if txns.empty:
            continue
        _, realised = replay(txns)
        txns["realised_pnl"] = [
            r if t == "SELL" else float("nan") for r, t in zip(realised, txns["tx_type"])
        ]
        frames.append(txns)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_day"] = out["ts"].dt.normalize()
    return out.sort_values(["_day", "id"], kind="stable").drop(columns="_day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Portfolio calculations
# ---------------------------------------------------------------------------

def _pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return numerator / denominator * 100


def compute_portfolio(conn: sqlite3.Connection, prices: dict) -> dict:
    """
    Per asset and in total: holdings, average cost, realised and unrealised profit in USD,
    and the same in rupiah when the USD/IDR rate loaded. Anything that needs a missing
    price or rate is None.
    """
    idr = prices.get("IDR_per_USD")

    result = {}
    for asset, price_key in [("BTC", "BTC_USD"), ("GOLD", "GOLD_USD_per_gram")]:
        pos, _ = replay(get_transactions(conn, asset))
        price = prices.get(price_key)
        if price is not None:
            value = pos.qty * price
        else:
            value = 0.0 if pos.qty == 0 else None  # nothing held: worth 0 whatever the price
        unrealised = value - pos.cost_basis if value is not None else None
        total_pnl = pos.realised_pnl + unrealised if unrealised is not None else None

        result[asset] = {
            "qty":                pos.qty,
            "avg_cost":           pos.avg_cost,
            "cost_basis":         pos.cost_basis,
            "invested":           pos.invested,
            "fees":               pos.fees,
            "current_price":      price,
            "current_value_usd":  value,
            "realised_pnl_usd":   pos.realised_pnl,
            "unrealised_pnl_usd": unrealised,
            "unrealised_pct":     _pct(unrealised, pos.cost_basis),
            "pnl_usd":            total_pnl,
            "pnl_pct":            _pct(total_pnl, pos.invested),
            "current_value_idr":  to_idr(value, idr),
            "pnl_idr":            to_idr(total_pnl, idr),
        }

    def total(key: str) -> Optional[float]:
        values = [result[a][key] for a in ASSETS]
        return None if any(v is None for v in values) else sum(values)

    total_value = total("current_value_usd")
    total_pnl = total("pnl_usd")
    result["TOTAL"] = {
        "cost_basis":         total("cost_basis"),
        "invested":           total("invested"),
        "fees":               total("fees"),
        "current_value_usd":  total_value,
        "realised_pnl_usd":   total("realised_pnl_usd"),
        "unrealised_pnl_usd": total("unrealised_pnl_usd"),
        "pnl_usd":            total_pnl,
        "pnl_pct":            _pct(total_pnl, total("invested")),
        "current_value_idr":  to_idr(total_value, idr),
        "pnl_idr":            to_idr(total_pnl, idr),
    }
    return result


def daily_pnl_series(conn: sqlite3.Connection, asset: str, hist_df: pd.DataFrame) -> pd.Series:
    """Total profit (realised + unrealised, USD) at each day's close, from the first trade on."""
    txns = get_transactions(conn, asset)
    if txns.empty or hist_df is None or hist_df.empty:
        return pd.Series(dtype=float)

    close = hist_df["Close"].dropna()
    days = pd.DatetimeIndex(pd.to_datetime(close.index))
    if days.tz is not None:
        days = days.tz_localize(None)
    days = days.normalize()
    trade_days = txns["ts"].dt.normalize()

    pos = Position()
    applied = 0
    points = {}
    for day, price in zip(days, close.to_numpy(dtype=float)):
        while applied < len(txns) and trade_days.iloc[applied] <= day:
            row = txns.iloc[applied]
            pos.apply(row["tx_type"], float(row["amount"]), float(row["price_usd"]), _fee(row["fee_usd"]))
            applied += 1
        if applied:
            points[day] = pos.realised_pnl + pos.qty * price - pos.cost_basis
    return pd.Series(points, dtype=float)
