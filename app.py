"""
app.py — Stoic Investor Dashboard
Tracks a Bitcoin and gold portfolio in USD and rupiah, shows a Prophet price forecast
next to its backtested track record, and picks a sourced Stoic quote for the day's
market mood.

Run:  streamlit run app.py
"""

import html
import logging
import time
import warnings
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Local modules ──────────────────────────────────────────────────────────
from data_engine import (
    init_db,
    fetch_all_prices,
    fetch_historical,
    add_transaction,
    compute_portfolio,
    daily_pnl_series,
    transactions_with_pnl,
)
from forecaster import PROPHET_AVAILABLE, forecast, forecast_summary, load_backtest, track_record
from stoic_search import STOIC_CORPUS, StoicSearch, attribution

MISSING = "—"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stoic Investor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --gold: #C9A84C;
    --gold-light: #E8C870;
    --btc: #F7931A;
    --green: #22c55e;
    --red: #ef4444;
    --bg: #0D0F14;
    --surface: #161A22;
    --border: #272D3A;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: #E2E8F0 !important;
}

h1, h2, h3 { font-family: 'Cinzel', serif; color: var(--gold) !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px !important;
}
div[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.75rem; }

/* Stoic quote card */
.stoic-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #0D1117 100%);
    border-left: 3px solid var(--gold);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-style: italic;
    color: #CBD5E1;
    line-height: 1.7;
}
.stoic-attribution {
    color: var(--gold);
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    margin-top: 8px;
    text-align: right;
    font-style: normal;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Tabs */
button[data-baseweb="tab"] { color: #94A3B8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%) !important;
    color: #0D0F14 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db():
    return init_db()

@st.cache_resource(show_spinner=False)
def get_search():
    return StoicSearch()

conn = get_db()
search = get_search()

# ── Data loaders with caching ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)   # 5-minute cache
def load_prices():
    return fetch_all_prices()

@st.cache_data(ttl=3600, show_spinner=False)  # 1-hour cache
def load_historical(asset: str, period: str = "1y"):
    return fetch_historical(asset, period)

@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast(asset: str, period: str = "1y", periods: int = 30):
    hist = load_historical(asset, period)
    if hist.empty:
        return None
    return forecast(hist, periods=periods)

@st.cache_data(show_spinner=False)
def get_backtest():
    return load_backtest()


# ===========================================================================
# HELPERS
# ===========================================================================

def fmt_usd(v, decimals: int = 2) -> str:
    if v is None:
        return MISSING
    sign, a = ("-" if v < 0 else ""), abs(v)
    if a >= 1_000_000:
        return f"{sign}${a/1_000_000:,.2f}M"
    return f"{sign}${a:,.{decimals}f}"

def fmt_idr(v) -> str:
    if v is None:
        return MISSING
    sign, a = ("-" if v < 0 else ""), abs(v)
    if a >= 1_000_000_000:
        return f"{sign}Rp {a/1_000_000_000:,.2f}B"
    if a >= 1_000_000:
        return f"{sign}Rp {a/1_000_000:,.1f}M"
    return f"{sign}Rp {a:,.0f}"

def fmt_pct(v) -> str:
    return MISSING if v is None else f"{v:+.1f}%"

def signed_usd(v) -> str:
    return MISSING if v is None else ("+" if v >= 0 else "") + fmt_usd(v)

def pnl_html(pnl, pct) -> str:
    if pnl is None:
        return "<span style='color:#94A3B8;'>Profit unavailable: the live price did not load</span>"
    color = "#22c55e" if pnl >= 0 else "#ef4444"
    return (
        f"<span style='color:{color}; font-size:1.1rem; font-weight:600;'>"
        f"{signed_usd(pnl)} ({fmt_pct(pct)})</span>"
    )

def stoic_html(quote: dict) -> str:
    return f"""
<div class="stoic-card">
    "{html.escape(quote['text'])}"
    <div class="stoic-attribution">— {html.escape(attribution(quote))}</div>
</div>"""


def build_price_chart(hist_df: pd.DataFrame, fcst_df, color: str) -> go.Figure:
    fig = go.Figure()

    # Historical candlestick
    fig.add_trace(go.Candlestick(
        x=hist_df.index,
        open=hist_df["Open"], high=hist_df["High"],
        low=hist_df["Low"],  close=hist_df["Close"],
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        name="Price",
        showlegend=False,
    ))

    # Forecast band, forecast line, and the "no change" line it is measured against
    if fcst_df is not None and not fcst_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fcst_df["ds"], fcst_df["ds"].iloc[::-1]]),
            y=pd.concat([fcst_df["yhat_upper"], fcst_df["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(201,168,76,0.12)",
            line=dict(width=0), name="Prophet 80% band", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=fcst_df["ds"], y=fcst_df["yhat"],
            line=dict(color=color, width=2, dash="dot"),
            name="Prophet forecast",
        ))
        last_close = float(hist_df["Close"].dropna().iloc[-1])
        fig.add_trace(go.Scatter(
            x=[fcst_df["ds"].iloc[0], fcst_df["ds"].iloc[-1]], y=[last_close, last_close],
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
            name="No change",
        ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#475569", rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="#1E2533", color="#475569"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        font=dict(color="#94A3B8"),
    )
    return fig


def build_pnl_chart(pnl: pd.Series, color: str) -> go.Figure:
    fig = go.Figure()
    fill_color = "rgba(34,197,94,0.15)" if pnl.iloc[-1] >= 0 else "rgba(239,68,68,0.15)"
    fig.add_trace(go.Scatter(
        x=pnl.index, y=pnl.values,
        fill="tozeroy", fillcolor=fill_color,
        line=dict(color=color, width=2),
        name="P&L (USD)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#475569"),
        yaxis=dict(showgrid=True, gridcolor="#1E2533", color="#475569"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        font=dict(color="#94A3B8"),
    )
    return fig


def show_track_record(asset: str, horizon: int, period: str):
    """What the backtest says about this exact forecast, right under it."""
    record = track_record(get_backtest(), asset, horizon, period)
    if record is None:
        if period != "1y":
            st.caption(
                "The backtest measures the default 1-year history only. "
                "Set History period to 1y to see how this forecast has done."
            )
        return
    worse = record["mape_model"] > record["mape_naive"]
    text = (
        f"**How far to trust this forecast.** Tested on {record['n']} past {horizon}-day forecasts "
        f"(one every 14 days, {record['first_origin'][:4]} to {record['last_origin'][:4]}), it missed "
        f"the real price by **{record['mape_model']:.1f}%** on average. Assuming the price would not "
        f"change missed by **{record['mape_naive']:.1f}%**. The forecast was closer {record['win_rate']:.0f}% "
        f"of the time, called the direction right {record['direction_hit_rate']:.0f}% of the time, and the "
        f"price ended inside its 80% band only {record['band_coverage']:.0f}% of the time."
    )
    if worse:
        st.warning(text + " Treat the line as a scenario, not a prediction.")
    else:
        st.success(text)


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("<h2 style='font-size:1.3rem;'>⚖️ Stoic Investor</h2>", unsafe_allow_html=True)
    st.caption("BTC & Gold Portfolio — IDR Edition")
    st.divider()

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        st.caption("⏱ Prices refresh every 5 minutes")

    st.divider()

    # ── Add Transaction ────────────────────────────────────────────────
    st.markdown("### ➕ Add Transaction")
    with st.form("add_transaction", clear_on_submit=True):
        tx_asset  = st.selectbox("Asset", ["BTC", "GOLD"])
        tx_type   = st.selectbox("Type", ["BUY", "SELL"])
        tx_date   = st.date_input("Trade date", value=date.today(), max_value=date.today())
        tx_amount = st.number_input(
            "Amount (BTC, or grams of gold)", min_value=0.0, step=0.001, format="%.6f"
        )
        tx_price  = st.number_input("Price per unit (USD per BTC, or per gram)", min_value=0.0, step=1.0)
        tx_fee    = st.number_input("Fee (USD)", min_value=0.0, step=0.5)
        tx_note   = st.text_input("Note (optional)")
        submitted = st.form_submit_button("Record Transaction", width="stretch")
    if submitted:
        try:
            add_transaction(
                conn, tx_asset, tx_type, tx_amount, tx_price,
                fee_usd=tx_fee, note=tx_note, ts=tx_date,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(f"✓ {tx_type} {tx_amount:g} {tx_asset} @ ${tx_price:,.2f} on {tx_date}")

    st.divider()

    # ── Forecast settings ───────────────────────────────────────────────
    st.markdown("### 🔮 Forecast Settings")
    forecast_period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=2)
    forecast_days   = st.slider("Forecast horizon (days)", 7, 90, 30)

    st.divider()
    st.caption("Data: Yahoo Finance · Forecast: Prophet · Quotes: public-domain translations")
    st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')}")


# ===========================================================================
# MAIN CONTENT
# ===========================================================================

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:2rem; letter-spacing:0.08em;'>⚖️ STOIC INVESTOR</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:0.85rem;'>Bitcoin & Gold — The Stoic Portfolio</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────
with st.spinner("Fetching live prices…"):
    prices = load_prices()

btc_price  = prices.get("BTC_USD")
gold_price = prices.get("GOLD_USD_per_gram")
idr_rate   = prices.get("IDR_per_USD")

missing = [name for name, value in [
    ("Bitcoin price", btc_price), ("gold price", gold_price), ("USD/IDR rate", idr_rate),
] if value is None]
if missing:
    st.warning(
        f"Could not load the {', '.join(missing)} from Yahoo Finance, so the values that need "
        f"{'it' if len(missing) == 1 else 'them'} show {MISSING} instead of a guess. "
        "Try again in a few minutes."
    )

try:
    portfolio = compute_portfolio(conn, prices)
except ValueError as exc:
    st.error(f"The transaction history does not add up: {exc}. Fix or delete that row in portfolio.db.")
    st.stop()

# ── Live price ticker ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "₿ Bitcoin (USD)",
        fmt_usd(btc_price, 0),
        help="BTC-USD via Yahoo Finance",
    )
with c2:
    st.metric(
        "🥇 Gold / gram (USD)",
        fmt_usd(gold_price, 2),
        help="COMEX gold futures (GC=F) ÷ 31.1035 g per troy oz. "
             "Indonesian retail gold (Antam, Pegadaian) costs more than this.",
    )
with c3:
    st.metric(
        "💵 USD / IDR",
        MISSING if idr_rate is None else f"Rp {idr_rate:,.0f}",
        help="IDR=X via Yahoo Finance",
    )
with c4:
    total = portfolio["TOTAL"]
    st.metric(
        "📊 Total P&L (USD)",
        signed_usd(total["pnl_usd"]),
        delta=None if total["pnl_pct"] is None else f"{total['pnl_pct']:+.1f}%",
        delta_color="normal",
        help="Realised profit from sales plus unrealised profit on what you still hold, "
             "after fees. The % is of everything you paid in.",
    )

st.divider()

# ── Stoic Oracle ─────────────────────────────────────────────────────────
# Market mood from Bitcoin's change between the last two daily closes
@st.cache_data(ttl=300)
def get_stoic_quote():
    try:
        hist = fetch_historical("BTC", "5d")
        if not hist.empty and len(hist) >= 2:
            yesterday = hist["Close"].iloc[-2]
            today     = hist["Close"].iloc[-1]
            pct_chg   = float((today - yesterday) / yesterday)
        else:
            pct_chg = None
    except Exception:
        pct_chg = None
    return search.context_for_change(pct_chg or 0.0), pct_chg

quote_result, pct_24h = get_stoic_quote()

if pct_24h is None:
    mood_label, change_label = "📊 Steady", "BTC daily change unavailable"
else:
    mood_label = "📉 Market Dip" if pct_24h < -0.05 else \
                 "🚀 Euphoria"   if pct_24h >  0.10 else \
                 "📊 Steady"
    change_label = f"BTC daily change: {pct_24h*100:+.1f}%"

st.markdown(f"**🏛 Stoic Oracle** · _{mood_label} · {change_label}_")
st.markdown(stoic_html(quote_result), unsafe_allow_html=True)
st.divider()

# ===========================================================================
# TABS
# ===========================================================================

tab_portfolio, tab_btc, tab_gold, tab_transactions, tab_quotes = st.tabs([
    "📊 Portfolio", "₿ Bitcoin", "🥇 Gold", "📋 Transactions", "🏛 Stoic Library",
])


# ── TAB 1: Portfolio Overview ─────────────────────────────────────────────
with tab_portfolio:
    st.markdown("### Portfolio Summary")
    col_btc, col_gold, col_total = st.columns(3)

    for col, asset, label in [
        (col_btc,   "BTC",  "₿ Bitcoin"),
        (col_gold,  "GOLD", "🥇 Gold"),
        (col_total, "TOTAL","📊 Total"),
    ]:
        with col:
            data = portfolio[asset]
            st.markdown(f"**{label}**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Value (USD)", fmt_usd(data["current_value_usd"]))
            with m2:
                st.metric("Value (IDR)", fmt_idr(data["current_value_idr"]))
            st.markdown(pnl_html(data["pnl_usd"], data["pnl_pct"]), unsafe_allow_html=True)
            st.caption(
                f"Realised: {signed_usd(data['realised_pnl_usd'])} · "
                f"Unrealised: {signed_usd(data['unrealised_pnl_usd'])}"
            )
            if asset != "TOTAL":
                unit = "BTC" if asset == "BTC" else "g"
                st.caption(f"Holdings: {data['qty']:.6g} {unit} · Avg cost: {fmt_usd(data['avg_cost'])} per {unit}")
            else:
                st.caption(f"Paid in: {fmt_usd(data['invested'])}, including {fmt_usd(data['fees'])} in fees")

    st.caption(
        "Average-cost method: a sale takes units out at their average cost, including buy fees. "
        "Its price minus that cost and its fee is realised profit; what you still hold, at today's "
        "price, minus what it cost is unrealised profit."
    )

    st.divider()
    st.markdown("### Portfolio Allocation")
    btc_val  = portfolio["BTC"]["current_value_usd"]
    gold_val = portfolio["GOLD"]["current_value_usd"]
    if btc_val is None or gold_val is None:
        st.info("Allocation needs both live prices.")
    elif btc_val + gold_val > 0:
        fig_pie = go.Figure(go.Pie(
            labels=["Bitcoin", "Gold"],
            values=[btc_val, gold_val],
            hole=0.5,
            marker_colors=["#F7931A", "#C9A84C"],
            textfont=dict(color="#E2E8F0"),
        ))
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True, height=300,
            legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pie, width="stretch", key="allocation")


# ── TABS 2 and 3: Bitcoin and Gold ────────────────────────────────────────
def asset_tab(asset: str, title: str, price, color: str, decimals: int, unit: str):
    st.markdown(f"### {title}")

    with st.spinner(f"Loading {asset} data…"):
        hist = load_historical(asset, forecast_period)
        fcst = load_forecast(asset, forecast_period, forecast_days)

    summ = forecast_summary(fcst, price)
    data = portfolio[asset]

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current price", fmt_usd(price, decimals))
    with m2: st.metric(f"{forecast_days}-day forecast", fmt_usd(summ["target_price"], decimals) if summ else MISSING)
    with m3: st.metric("Forecast change", fmt_pct(summ["change_pct"]) if summ else MISSING)
    with m4: st.metric("Holdings", f"{data['qty']:.6g} {unit}")

    st.markdown("**Price chart and forecast**")
    if hist.empty:
        st.warning(f"Could not load {asset} price history. Check your connection.")
        return
    st.plotly_chart(build_price_chart(hist, fcst, color), width="stretch", key=f"{asset}_price")

    if fcst is None:
        st.info("Forecasts need the prophet package (pip install prophet)." if not PROPHET_AVAILABLE
                else "Not enough price history for a forecast.")
    else:
        show_track_record(asset, forecast_days, forecast_period)
        with st.expander("📊 Forecast details"):
            fa, fb, fc = st.columns(3)
            fa.metric("Target (low)",  fmt_usd(summ["target_lower"], decimals))
            fb.metric("Target (mid)",  fmt_usd(summ["target_price"], decimals))
            fc.metric("Target (high)", fmt_usd(summ["target_upper"], decimals))
            st.caption(f"Prophet's 80% band for {summ['target_date']:%d %b %Y}. "
                       "The track record above says how often the real price ended inside it.")

    st.markdown("**Profit over time (realised + unrealised)**")
    pnl = daily_pnl_series(conn, asset, hist)
    if pnl.empty:
        st.caption("No trades in this period yet.")
    else:
        st.plotly_chart(build_pnl_chart(pnl, color), width="stretch", key=f"{asset}_pnl")


with tab_btc:
    asset_tab("BTC", "₿ Bitcoin Analysis", btc_price, "#F7931A", 0, "BTC")

with tab_gold:
    asset_tab("GOLD", "🥇 Gold Analysis (USD per gram)", gold_price, "#C9A84C", 2, "g")


# ── TAB 4: Transactions ───────────────────────────────────────────────────
with tab_transactions:
    st.markdown("### 📋 Transaction History")
    df_tx = transactions_with_pnl(conn)
    if df_tx.empty:
        st.info("No transactions yet. Add one in the sidebar.")
    else:
        display_df = df_tx.copy()
        display_df["ts"] = display_df["ts"].dt.strftime("%Y-%m-%d")
        display_df["value_usd"] = display_df["amount"] * display_df["price_usd"]
        st.dataframe(
            display_df[[
                "ts", "asset", "tx_type", "amount", "price_usd",
                "fee_usd", "value_usd", "realised_pnl", "note",
            ]].rename(columns={
                "ts": "Date", "asset": "Asset", "tx_type": "Type",
                "amount": "Qty", "price_usd": "Price (USD)", "fee_usd": "Fee (USD)",
                "value_usd": "Value (USD)", "realised_pnl": "Realised P&L (USD)", "note": "Note",
            }),
            width="stretch",
            hide_index=True,
        )
        total = portfolio["TOTAL"]
        st.caption(
            f"Paid in: **{fmt_usd(total['invested'])}** · Realised P&L: **{signed_usd(total['realised_pnl_usd'])}** "
            f"· Fees: **{fmt_usd(total['fees'])}** · {len(df_tx)} transactions. "
            "Trades on the same day are applied in the order they were entered."
        )


# ── TAB 5: Stoic Library ─────────────────────────────────────────────────
with tab_quotes:
    st.markdown("### 🏛 Stoic Wisdom Library")
    if search.mode == "semantic":
        st.caption(
            f"Semantic search over {len(STOIC_CORPUS)} quotes: ChromaDB finds the ones closest in meaning "
            "to what you describe (all-MiniLM-L6-v2 sentence embeddings). Nothing is generated. Every "
            "quote is word for word from a public-domain translation, with its citation."
        )
    else:
        st.caption(
            f"Keyword search over {len(STOIC_CORPUS)} quotes (ChromaDB is unavailable here). Every "
            "quote is word for word from a public-domain translation, with its citation."
        )

    situation_input = st.text_input(
        "Describe your market situation",
        placeholder="e.g. Bitcoin dropped 20% today and I want to sell everything",
    )

    col_auto, col_search = st.columns([1, 1])
    with col_auto:
        if st.button("🔍 Find Relevant Quotes", width="stretch"):
            if situation_input:
                for r in search.query(situation_input, n_results=3):
                    st.markdown(stoic_html(r), unsafe_allow_html=True)
            else:
                st.warning("Describe a situation first.")

    with col_search:
        preset = st.selectbox("Or pick a scenario:", [
            "— select —",
            "Market crash / dip",
            "FOMO - missing a pump",
            "Panic selling urge",
            "Taking profits greed",
            "DCA discipline",
            "Long-term patience",
        ])
        if preset != "— select —" and st.button("Apply Scenario", width="stretch"):
            for r in search.query(preset, n_results=3):
                st.markdown(stoic_html(r), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📚 All quotes")
    search_filter = st.text_input("Filter by author, work or keyword", placeholder="e.g. Seneca")
    needle = search_filter.lower()
    filtered = [
        q for q in STOIC_CORPUS
        if not needle
        or needle in q["author"].lower()
        or needle in q["citation"].lower()
        or needle in q["text"].lower()
        or needle in q["tags"].lower()
    ]
    st.caption(f"Showing {len(filtered)} / {len(STOIC_CORPUS)} quotes")
    for q in filtered:
        with st.expander(f"{q['author']}, {q['citation']} — {q['text'][:70]}…"):
            st.markdown(f"> *{q['text']}*")
            st.caption(f"**{attribution(q)}** · [source text]({q['source_url']})")
            st.caption(f"Tags: `{q['tags']}`")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#334155; font-size:0.75rem;'>"
    "Stoic Investor · Prices via Yahoo Finance · "
    "Forecast via Prophet, backtested against a no-change guess · "
    "Quotes from public-domain translations · Not financial advice."
    "</p>",
    unsafe_allow_html=True,
)

# ── Auto-refresh ───────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
