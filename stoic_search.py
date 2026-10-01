"""
stoic_search.py
Semantic search over the sourced Stoic quotes in stoic_quotes.json.

Every quote is word for word from a public-domain translation and carries its citation
(docs/quote-audit.md says how the list was checked). Search finds the quotes closest in
meaning to a market situation, using ChromaDB's built-in sentence embeddings
(all-MiniLM-L6-v2). Nothing is generated: the app only ever shows a real quote. When
ChromaDB or its embedding model is unavailable, it falls back to matching keywords.
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

QUOTES_PATH = Path(__file__).resolve().parent / "stoic_quotes.json"
COLLECTION_PREFIX = "stoic_quotes_"
# The first version stored its unsourced quotes here; remove it so they can never come back.
LEGACY_COLLECTIONS = ("stoic_wisdom",)


def load_quotes(path: Path = QUOTES_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


STOIC_CORPUS = load_quotes()


def attribution(quote: dict) -> str:
    """e.g. 'Marcus Aurelius, Meditations 8.47 (tr. George Long, 1862)'."""
    return (
        f"{quote['author']}, {quote['citation']} "
        f"(tr. {quote['translator']}, {quote['translation_year']})"
    )


def fingerprint(quotes: list) -> str:
    """Changes whenever a quote is added, removed or reworded."""
    blob = json.dumps([[q["id"], q["text"]] for q in quotes], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


CHROMA_MIN_SQLITE = (3, 35, 0)


def use_bundled_sqlite_if_needed() -> bool:
    """
    ChromaDB needs SQLite 3.35 or newer, and some Linux hosts ship an older one: Streamlit
    Community Cloud runs Debian 11, with SQLite 3.34. There, requirements.txt installs
    pysqlite3-binary, whose bundled SQLite is newer, and this puts it in place of the
    standard sqlite3 module before ChromaDB is imported. Returns True when it did.
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= CHROMA_MIN_SQLITE:
        return False
    try:
        import pysqlite3  # type: ignore  # noqa: F401
    except ImportError:
        return False
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    return True


class StoicSearch:
    """Finds the quotes that best fit a situation; `mode` is 'semantic' or 'keyword'."""

    def __init__(self, persist_dir: str = ".chromadb", quotes: Optional[list] = None):
        self.quotes = quotes if quotes is not None else STOIC_CORPUS
        self._by_id = {q["id"]: q for q in self.quotes}
        self._collection = None
        self.mode = "keyword"
        self._init_chroma(persist_dir)

    def _init_chroma(self, persist_dir: str) -> None:
        try:
            use_bundled_sqlite_if_needed()
            import chromadb
            from chromadb.utils import embedding_functions

            client = chromadb.PersistentClient(path=persist_dir)
            # One collection per version of the quote list, so an edit is never served stale.
            name = COLLECTION_PREFIX + fingerprint(self.quotes)
            for col in client.list_collections():
                col_name = getattr(col, "name", col)
                if col_name != name and (
                    col_name.startswith(COLLECTION_PREFIX) or col_name in LEGACY_COLLECTIONS
                ):
                    client.delete_collection(col_name)
            collection = client.get_or_create_collection(
                name=name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )
            if collection.count() != len(self.quotes):
                collection.upsert(
                    ids=[q["id"] for q in self.quotes], documents=[q["text"] for q in self.quotes]
                )
            self._collection = collection
            self.mode = "semantic"
            logger.info("StoicSearch: %d quotes indexed in ChromaDB", len(self.quotes))
        except Exception as exc:  # noqa: BLE001 - any failure means the keyword fallback
            logger.warning("StoicSearch: ChromaDB unavailable (%s), matching keywords instead", exc)

    # ------------------------------------------------------------------
    def query(self, situation: str, n_results: int = 1) -> list:
        """The n quotes closest in meaning to `situation`: quote dicts plus 'distance'."""
        n = max(1, min(int(n_results), len(self.quotes)))
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[situation], n_results=n)
                return [
                    {**self._by_id[qid], "distance": dist}
                    for qid, dist in zip(res["ids"][0], res["distances"][0])
                ]
            except Exception as exc:  # noqa: BLE001
                logger.error("StoicSearch: query failed (%s), matching keywords instead", exc)
        return self._keyword_query(situation, n)

    def _keyword_query(self, situation: str, n: int) -> list:
        tokens = set(re.findall(r"[a-z]+", situation.lower()))

        def overlap(q: dict) -> int:
            words = set(q["tags"].lower().replace("_", " ").split())
            words |= {w for w in re.findall(r"[a-z]+", q["text"].lower()) if len(w) > 3}
            return len(tokens & words)

        ranked = sorted(self.quotes, key=overlap, reverse=True)  # ties keep the file's order
        return [{**q, "distance": None} for q in ranked[:n]]

    # ------------------------------------------------------------------
    # Market moods
    # ------------------------------------------------------------------
    def on_market_dip(self, pct_change: float) -> dict:
        """pct_change is negative (e.g. -0.07 for -7%)."""
        situation = "severe crash" if pct_change < -0.15 else "market dip loss fear"
        return self.query(situation)[0]

    def on_euphoria(self, pct_change: float) -> dict:
        return self.query("greed euphoria all time high wealth")[0]

    def on_fomo(self) -> dict:
        return self.query("FOMO crowd herd mentality")[0]

    def on_neutral(self) -> dict:
        return self.query("calm neutral patience discipline")[0]

    def context_for_change(self, pct_change: float) -> dict:
        """The quote for a day's price change (a fraction: -0.07 is -7%)."""
        if pct_change < -0.05:
            return self.on_market_dip(pct_change)
        if pct_change > 0.10:
            return self.on_euphoria(pct_change)
        if -0.02 <= pct_change <= 0.02:
            return self.on_neutral()
        return self.query("patience resilience")[0]
