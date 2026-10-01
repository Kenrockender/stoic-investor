# ⚖️ Stoic Investor
> *"If thou art pained by any external thing, it is not this thing that disturbs thee, but thy own judgment about it."*
> — Marcus Aurelius, Meditations 8.47 (tr. George Long)

A Streamlit dashboard for a Bitcoin and gold portfolio, in US dollars and rupiah. It tracks what you bought and sold, shows a price forecast together with how often that forecast has been wrong, and picks a Stoic quote to match the day's market mood.

## What it found: the forecast loses to "no change"

The app forecasts with Prophet. `backtest.py` replays that exact forecast every 14 days from September 2015 to June 2026, 282 times per asset, each time using only the year of prices before that day. It then compares each forecast with what the price actually did, and with the simplest possible guess: that the price stays where it is.

| | Horizon | Prophet's average miss | "No change" average miss | Prophet closer | Direction right | Price inside the 80% band |
|---|---|---|---|---|---|---|
| Bitcoin | 7 days | 9.7% | 6.1% | 36% | 54% | 37% |
| Bitcoin | 30 days | 24.2% | 15.0% | 37% | 50% | 17% |
| Bitcoin | 90 days | 47.0% | 28.4% | 35% | 42% | 6% |
| Gold | 7 days | 2.7% | 1.6% | 31% | 50% | 28% |
| Gold | 30 days | 7.1% | 3.3% | 30% | 51% | 12% |
| Gold | 90 days | 15.7% | 6.3% | 38% | 55% | 9% |

At every horizon and for both assets, assuming no change beat the forecast. The forecast called the direction about as often as a coin toss, and its "80%" band contained the real price far less often than 80% of the time. So the app shows each forecast with its track record and a "no change" line, and calls it a scenario rather than a prediction. Full results for every horizon from 1 to 90 days are in `results/backtest.json`.

## What it does

- **Portfolio:** live BTC, gold and USD/IDR prices from Yahoo Finance. Profit uses the average-cost method: a sale takes units out at their average cost, including buy fees. Its price minus that cost and its fee is realised profit, and what you still hold, at today's price, minus what it cost is unrealised profit.
- **No guessed numbers:** if a price or the exchange rate fails to load, the app says so and shows "—". It never falls back to a made-up rate.
- **Transactions:** buys and sells with a trade date and fee. A sale of more than you held on that date is refused, including a back-dated sale that would make a later one impossible.
- **Forecast:** Prophet with an 80% band, for 7 to 90 days ahead, next to its backtested track record.
- **Stoic quotes:** semantic search (ChromaDB with all-MiniLM-L6-v2 sentence embeddings) over 32 quotes. Nothing is generated: every quote is word for word from a public-domain translation and shows its citation, for example "Meditations 8.47". [docs/quote-audit.md](docs/quote-audit.md) explains how the original 40 quotes were checked. Several were modern paraphrases, duplicates or misattributions.

## Quick start

With conda, for example an environment called `finance_project`:

```bash
conda activate finance_project      # activation tells Prophet where its Stan engine is
pip install -r requirements.txt
streamlit run app.py
```

With a plain virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 and needs no API keys. On the first run it creates `portfolio.db` with four demo purchases (125 g of gold and 0.10 BTC). Add your own trades from the sidebar, or delete `portfolio.db` to start empty.

## Tests

```bash
pip install -r requirements-dev.txt
python scripts/download_quote_sources.py   # optional: lets the tests check every quote word for word
pytest
```

The 65 tests cover the profit maths (including the partial-sale and fee cases the first version got wrong), the oversell checks, missing prices, the forecast and the backtest's scoring rules, the quote list, the quote search, and a run of the whole app on fake prices.

## Re-running the backtest

```bash
python backtest.py      # downloads daily prices into data/prices.csv, then about 2 minutes on 8 cores
```

Yahoo Finance's terms allow its data for personal use, so the price file is not in this repository. Only the derived error statistics are.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `data_engine.py` | Yahoo Finance prices, the SQLite transaction store, average-cost profit |
| `forecaster.py` | The Prophet forecast and its track record from the backtest |
| `backtest.py` | Scores the forecast against "no change" and writes `results/` |
| `stoic_search.py` | Semantic search over `stoic_quotes.json` |
| `stoic_quotes.json` | 32 sourced quotes with citations |
| `docs/quote-audit.md` | What happened to each original quote |
| `tests/` | pytest suite |

## Data notes

- Gold is priced from COMEX gold futures (GC=F) per troy ounce, converted to grams. Indonesian retail gold (Antam, Pegadaian) costs more.
- Prices come from Yahoo Finance through `yfinance`, which is unofficial and occasionally fails. When it does, the app shows "—" instead of a number.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On [share.streamlit.io](https://share.streamlit.io), create an app that points to `app.py`.
3. `requirements.txt` installs everything. Prophet's install can take a few minutes on the first start.

## Disclaimer

This is a personal portfolio tracker. Nothing here is financial advice.
