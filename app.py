"""
app.py — Stoic Investor Dashboard
A Compound AI System for BTC & Gold portfolio tracking with
time-series forecasting and Stoic philosophy RAG.

Run:  streamlit run app.py
"""

import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

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
    get_transactions,
    add_transaction,
    compute_portfolio,
    daily_pnl_series,
    GOLD_GRAMS_PER_OZ,
)
from forecaster import forecast, forecast_summary
from stoic_rag import StoicRAG

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
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%) !important;
    color: #0D0F14 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox select {
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
def get_rag():
    return StoicRAG()

conn = get_db()
rag  = get_rag()

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
    return forecast(hist, periods=periods, asset_name=asset)


# ===========================================================================
# HELPERS
# ===========================================================================

def fmt_usd(v: float, decimals: int = 2) -> str:
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v:,.{decimals}f}"
    return f"${v:.{decimals}f}"

def fmt_idr(v: float) -> str:
    if abs(v) >= 1_000_000_000:
        return f"Rp {v/1_000_000_000:,.2f}B"
    if abs(v) >= 1_000_000:
        return f"Rp {v/1_000_000:,.1f}M"
    return f"Rp {v:,.0f}"

def pnl_color(v: float) -> str:
    return "#22c55e" if v >= 0 else "#ef4444"

def delta_arrow(v: float) -> str:
    return "▲" if v >= 0 else "▼"


def stoic_html(quote: dict) -> str:
    return f"""
<div class="stoic-card">
    "{quote['text']}"
    <div class="stoic-attribution">— {quote['author']}, <em>{quote['source']}</em></div>
</div>"""


def build_price_chart(hist_df: pd.DataFrame, fcst_df: pd.DataFrame,
                      asset: str, color: str) -> go.Figure:
    fig = go.Figure()

    # Historical candlestick
    if not hist_df.empty:
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"], high=hist_df["High"],
            low=hist_df["Low"],  close=hist_df["Close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
            name="Price",
            showlegend=False,
        ))

    # Forecast band
    if fcst_df is not None and not fcst_df.empty:
        future = fcst_df[fcst_df["ds"] > hist_df.index.max()] if not hist_df.empty else fcst_df
        if not future.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
                y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(201,168,76,0.12)",
                line=dict(width=0), name="80% CI", showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=future["ds"], y=future["yhat"],
                line=dict(color=color, width=2, dash="dot"),
                name="Forecast",
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


def build_pnl_chart(conn, asset: str, hist_df: pd.DataFrame, color: str) -> go.Figure:
    pnl = daily_pnl_series(conn, asset, hist_df)
    fig = go.Figure()
    if pnl.empty:
        return fig
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
    tx_asset   = st.selectbox("Asset",   ["BTC", "GOLD"])
    tx_type    = st.selectbox("Type",    ["BUY", "SELL"])
    tx_amount  = st.number_input(
        "Amount (BTC units | grams)", min_value=0.0, step=0.001, format="%.6f"
    )
    tx_price   = st.number_input("Price (USD)", min_value=0.0, step=1.0)
    tx_note    = st.text_input("Note (optional)")

    if st.button("Record Transaction", use_container_width=True):
        if tx_amount > 0 and tx_price > 0:
            add_transaction(conn, tx_asset, tx_type, tx_amount, tx_price, note=tx_note)
            st.success(f"✓ {tx_type} {tx_amount} {tx_asset} @ ${tx_price:,.2f}")
            st.cache_data.clear()
        else:
            st.warning("Enter a valid amount and price.")

    st.divider()

    # ── Forecast settings ───────────────────────────────────────────────
    st.markdown("### 🔮 Forecast Settings")
    forecast_period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=2)
    forecast_days   = st.slider("Forecast horizon (days)", 7, 90, 30)

    st.divider()
    st.caption("Data: Yahoo Finance · Forecast: Prophet · Wisdom: Stoic Corpus")
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

idr_rate   = prices.get("IDR_per_USD") or 15_800
btc_price  = prices.get("BTC_USD") or 0
gold_price = prices.get("GOLD_USD_per_gram") or 0
portfolio  = compute_portfolio(conn, prices)

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
        help="GC=F futures ÷ 31.1g per troy oz",
    )
with c3:
    st.metric(
        "💵 USD / IDR",
        f"Rp {idr_rate:,.0f}",
        help="IDR=X via Yahoo Finance",
    )
with c4:
    total = portfolio.get("TOTAL", {})
    pnl   = total.get("pnl_usd", 0)
    pct   = total.get("pnl_pct", 0)
    color = "#22c55e" if pnl >= 0 else "#ef4444"
    st.metric(
        "📊 Total P&L (USD)",
        fmt_usd(pnl),
        delta=f"{pct:+.1f}%",
        delta_color="normal",
    )

st.divider()

# ── Stoic Oracle ─────────────────────────────────────────────────────────
# Determine market mood from 24h BTC change proxy (use forecast direction)
@st.cache_data(ttl=300)
def get_stoic_quote(btc_p: float) -> dict:
    # Use today vs yesterday from historical data as mood signal
    try:
        hist = fetch_historical("BTC", "5d")
        if not hist.empty and len(hist) >= 2:
            yesterday = hist["Close"].iloc[-2]
            today     = hist["Close"].iloc[-1]
            pct_chg   = (today - yesterday) / yesterday
        else:
            pct_chg = 0.0
    except Exception:
        pct_chg = 0.0
    return rag.context_for_change(pct_chg), pct_chg

quote_result, pct_24h = get_stoic_quote(btc_price)

mood_label = "📉 Market Dip" if pct_24h < -0.05 else \
             "🚀 Euphoria"   if pct_24h >  0.10 else \
             "📊 Steady"

st.markdown(f"**🏛 Stoic Oracle** · _{mood_label} · BTC 24h: {pct_24h*100:+.1f}%_")
st.markdown(stoic_html(quote_result), unsafe_allow_html=True)
st.divider()

# ===========================================================================
# TABS
# ===========================================================================

tab_portfolio, tab_btc, tab_gold, tab_transactions, tab_rag = st.tabs([
    "📊 Portfolio", "₿ Bitcoin", "🥇 Gold", "📋 Transactions", "🏛 Stoic Library",
])


# ── TAB 1: Portfolio Overview ─────────────────────────────────────────────
with tab_portfolio:
    st.markdown("### Portfolio Summary")
    col_btc, col_gold, col_total = st.columns(3)

    for col, asset, label, color in [
        (col_btc,   "BTC",  "₿ Bitcoin", "#F7931A"),
        (col_gold,  "GOLD", "🥇 Gold",   "#C9A84C"),
        (col_total, "TOTAL","📊 Total",  "#8B9BC8"),
    ]:
        with col:
            data = portfolio.get(asset, {})
            val  = data.get("current_value_usd", 0)
            pnl  = data.get("pnl_usd", 0)
            pct  = data.get("pnl_pct", 0)
            val_idr = data.get("current_value_idr", 0)

            st.markdown(f"**{label}**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Value (USD)", fmt_usd(val))
            with m2:
                st.metric("Value (IDR)", fmt_idr(val_idr))
            sign = "+" if pnl >= 0 else ""
            pnl_c = "#22c55e" if pnl >= 0 else "#ef4444"
            st.markdown(
                f"<span style='color:{pnl_c}; font-size:1.1rem; font-weight:600;'>"
                f"{sign}{fmt_usd(pnl)} ({sign}{pct:.1f}%)</span>",
                unsafe_allow_html=True,
            )
            if asset != "TOTAL":
                qty  = data.get("qty", 0)
                avg  = data.get("avg_cost", 0)
                unit = "BTC" if asset == "BTC" else "g"
                st.caption(f"Holdings: {qty:.6g} {unit} · Avg cost: {fmt_usd(avg)}")

    st.divider()
    st.markdown("### Portfolio Allocation")
    btc_val  = portfolio.get("BTC",  {}).get("current_value_usd", 0)
    gold_val = portfolio.get("GOLD", {}).get("current_value_usd", 0)
    if btc_val + gold_val > 0:
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
        st.plotly_chart(fig_pie, use_container_width=True)


# ── TAB 2: Bitcoin ────────────────────────────────────────────────────────
with tab_btc:
    st.markdown("### ₿ Bitcoin Analysis")

    with st.spinner("Loading BTC data…"):
        hist_btc = load_historical("BTC", forecast_period)
        fcst_btc = load_forecast("BTC", forecast_period, forecast_days)

    summ = forecast_summary(fcst_btc, btc_price) if fcst_btc is not None else {}

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current Price", fmt_usd(btc_price, 0))
    with m2: st.metric("30d Forecast", fmt_usd(summ.get("target_price", 0), 0) if summ else "—")
    with m3: st.metric("Expected Δ", f"{summ.get('change_pct', 0):+.1f}%" if summ else "—")
    with m4:
        btc_data = portfolio.get("BTC", {})
        st.metric("Holdings", f"{btc_data.get('qty', 0):.6f} BTC")

    st.markdown("**Price Chart + Forecast**")
    if not hist_btc.empty:
        st.plotly_chart(
            build_price_chart(hist_btc, fcst_btc, "BTC", "#F7931A"),
            use_container_width=True,
        )
    else:
        st.warning("Could not load BTC price data. Check your connection.")

    st.markdown("**Unrealised P&L over time**")
    if not hist_btc.empty:
        st.plotly_chart(build_pnl_chart(conn, "BTC", hist_btc, "#F7931A"), use_container_width=True)

    if summ:
        with st.expander("📊 Forecast Details"):
            fa, fb, fc = st.columns(3)
            fa.metric("Target (low)",  fmt_usd(summ["target_lower"], 0))
            fb.metric("Target (mid)",  fmt_usd(summ["target_price"], 0))
            fc.metric("Target (high)", fmt_usd(summ["target_upper"], 0))
            st.caption("80% confidence interval based on historical volatility.")


# ── TAB 3: Gold ───────────────────────────────────────────────────────────
with tab_gold:
    st.markdown("### 🥇 Gold Analysis")

    with st.spinner("Loading Gold data…"):
        hist_gold = load_historical("GOLD", forecast_period)
        fcst_gold = load_forecast("GOLD", forecast_period, forecast_days)

    summ_g = forecast_summary(fcst_gold, gold_price) if fcst_gold is not None else {}
    gold_data = portfolio.get("GOLD", {})

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current (USD/g)", fmt_usd(gold_price))
    with m2: st.metric("30d Forecast",   fmt_usd(summ_g.get("target_price", 0)) if summ_g else "—")
    with m3: st.metric("Expected Δ",     f"{summ_g.get('change_pct', 0):+.1f}%" if summ_g else "—")
    with m4: st.metric("Holdings",       f"{gold_data.get('qty', 0):.2f} g")

    st.markdown("**Price Chart + Forecast**")
    if not hist_gold.empty:
        st.plotly_chart(
            build_price_chart(hist_gold, fcst_gold, "GOLD", "#C9A84C"),
            use_container_width=True,
        )
    else:
        st.warning("Could not load Gold price data.")

    st.markdown("**Unrealised P&L over time**")
    if not hist_gold.empty:
        st.plotly_chart(build_pnl_chart(conn, "GOLD", hist_gold, "#C9A84C"), use_container_width=True)

    if summ_g:
        with st.expander("📊 Forecast Details"):
            fa, fb, fc = st.columns(3)
            fa.metric("Target (low)",  fmt_usd(summ_g["target_lower"]))
            fb.metric("Target (mid)",  fmt_usd(summ_g["target_price"]))
            fc.metric("Target (high)", fmt_usd(summ_g["target_upper"]))


# ── TAB 4: Transactions ───────────────────────────────────────────────────
with tab_transactions:
    st.markdown("### 📋 Transaction History")
    df_tx = get_transactions(conn)
    if df_tx.empty:
        st.info("No transactions yet. Add one in the sidebar.")
    else:
        # Display enriched table
        display_df = df_tx.copy()
        display_df["ts"] = display_df["ts"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["value_usd"] = display_df["amount"] * display_df["price_usd"]
        # Current P&L per tx row (mark-to-market)
        def mtm(row):
            cur = btc_price if row["asset"] == "BTC" else gold_price
            if row["tx_type"] == "BUY":
                return (cur - row["price_usd"]) * row["amount"]
            return 0.0
        display_df["mtm_pnl"] = display_df.apply(mtm, axis=1)

        st.dataframe(
            display_df[[
                "ts", "asset", "tx_type", "amount",
                "price_usd", "value_usd", "mtm_pnl", "note",
            ]].rename(columns={
                "ts": "Date", "asset": "Asset", "tx_type": "Type",
                "amount": "Qty", "price_usd": "Price (USD)",
                "value_usd": "Value (USD)", "mtm_pnl": "MTM P&L", "note": "Note",
            }),
            use_container_width=True,
            hide_index=True,
        )

        total_invested = (
            df_tx[df_tx["tx_type"] == "BUY"]["amount"]
            * df_tx[df_tx["tx_type"] == "BUY"]["price_usd"]
        ).sum()
        st.caption(f"Total invested: **{fmt_usd(total_invested)}** across {len(df_tx)} transactions")


# ── TAB 5: Stoic Library ─────────────────────────────────────────────────
with tab_rag:
    st.markdown("### 🏛 Stoic Wisdom Library")
    st.markdown("Ask the Stoics for guidance on any market scenario.")

    situation_input = st.text_input(
        "Describe your market situation",
        placeholder="e.g. Bitcoin dropped 20% today and I want to sell everything",
    )

    col_auto, col_search = st.columns([1, 1])
    with col_auto:
        if st.button("🔍 Find Relevant Quote", use_container_width=True):
            if situation_input:
                results = rag.query(situation_input, n_results=3)
                for r in results:
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
        if preset != "— select —" and st.button("Apply Scenario", use_container_width=True):
            results = rag.query(preset, n_results=3)
            for r in results:
                st.markdown(stoic_html(r), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📚 Full Corpus")
    from stoic_rag import STOIC_CORPUS
    search_filter = st.text_input("Filter by author or keyword", placeholder="e.g. Seneca")
    filtered = [
        q for q in STOIC_CORPUS
        if not search_filter
        or search_filter.lower() in q["author"].lower()
        or search_filter.lower() in q["text"].lower()
        or search_filter.lower() in q["tags"].lower()
    ]
    st.caption(f"Showing {len(filtered)} / {len(STOIC_CORPUS)} quotes")
    for q in filtered:
        with st.expander(f"{q['author']} — {q['text'][:80]}…"):
            st.markdown(f"> *{q['text']}*")
            st.caption(f"**{q['author']}**, {q['source']}")
            st.caption(f"Tags: `{q['tags']}`")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#334155; font-size:0.75rem;'>"
    "Stoic Investor · Data via Yahoo Finance · "
    "Forecasting via Prophet/NeuralProphet · Wisdom via Stoic Corpus · "
    "Not financial advice."
    "</p>",
    unsafe_allow_html=True,
)

# ── Auto-refresh ───────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
