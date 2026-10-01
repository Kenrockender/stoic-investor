# ⚖️ Stoic Investor
> *"You have power over your mind, not outside events. Realize this, and you will find strength."*
> — Marcus Aurelius

A **Compound AI System** for tracking your Bitcoin and Gold portfolio with time-series forecasting and a Stoic philosophy RAG engine that keeps you calm when the market bleeds.

---

## Architecture

```
stoic_investor/
├── app.py           ← Streamlit dashboard (UI layer)
├── data_engine.py   ← Live prices (yfinance) + SQLite transaction store
├── forecaster.py    ← Prophet / NeuralProphet time-series forecasting
├── stoic_rag.py     ← ChromaDB vector store with 40 curated Stoic quotes
├── requirements.txt
└── README.md

Runtime artefacts (auto-created):
├── portfolio.db     ← SQLite — your transaction history
└── .chromadb/       ← ChromaDB persistence directory
```

### The Four Pillars

| Layer | Tech | Purpose |
|---|---|---|
| **Data Engine** | `yfinance` + `sqlite3` | Live BTC/Gold/IDR prices, transaction history, P&L |
| **Forecaster** | `Prophet` / `NeuralProphet` | 7–90 day price forecasts with confidence bands |
| **Stoic RAG** | `ChromaDB` + sentence-transformers | Semantic retrieval of philosophy on market events |
| **UI** | `Streamlit` + `Plotly` | Dark-mode dashboard, candlestick charts, portfolio view |

---

## Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/yourname/stoic-investor
cd stoic-investor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on Prophet:** If `pip install prophet` fails, try:
> ```bash
> conda install -c conda-forge prophet
> ```
> The app gracefully falls back to a built-in naive trend model if neither
> Prophet nor NeuralProphet is available.

### 2 — Run

```bash
streamlit run app.py
```

Open http://localhost:8501 — that's it. No API keys required.

---

## Features

### 📊 Portfolio Dashboard
- Real-time BTC, Gold (per gram), and USD/IDR rates via Yahoo Finance
- Cost-basis tracking with average entry price per asset
- Unrealised P&L in both USD and IDR
- Portfolio allocation donut chart

### ₿ Bitcoin & 🥇 Gold Tabs
- Interactive candlestick charts (1y default, configurable)
- Prophet forecast overlay with 80% confidence band
- Daily P&L area chart

### 🏛 Stoic Oracle
- Automatically detects market mood (dip / euphoria / neutral) from 24h price change
- Retrieves the most semantically relevant Stoic quote using ChromaDB cosine similarity
- Works even offline (keyword fallback)

### 📋 Transactions
- Add BUY/SELL from the sidebar
- Mark-to-market P&L per transaction
- Pre-seeded with demo holdings (125g gold + 0.10 BTC via DCA)

### 🏛 Stoic Library
- Full-text search across 40 curated quotes (Marcus Aurelius, Seneca, Epictetus)
- Semantic query: describe any situation, get the most relevant wisdom
- Pre-set scenarios: crash, FOMO, panic selling, DCA discipline

---

## Customising Your Holdings

The database is seeded with demo transactions on first run:
- 125g of Gold at various prices
- 0.10 BTC via DCA

To replace with your real data, either:
1. Use the **sidebar form** to add real transactions, or
2. Delete `portfolio.db` and re-seed via the form

---

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to `app.py`
4. Add `requirements.txt` (already included)

> Prophet may take ~3 min to install on first cold start. The app will use
> the naive trend model in the meantime.

---

## Roadmap

- [ ] Oanda API integration for real-time gold spot price
- [ ] Push alerts (Telegram / email) on >5% daily moves
- [ ] DCA calculator with optimal entry simulator
- [ ] Multi-currency IDR/SGD/USD toggle

---

## Disclaimer

This is a personal portfolio tracker. Nothing here constitutes financial advice.
The Stoics would agree: the only thing you truly own is your response to events.
