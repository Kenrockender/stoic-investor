"""
The quote list: every entry is complete and distinct, and, when the source texts have been
downloaded (python scripts/download_quote_sources.py), every quote appears word for word
in the public-domain translation it cites.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUOTES = json.loads((ROOT / "stoic_quotes.json").read_text(encoding="utf-8"))
SOURCES = ROOT / ".quote_sources"
KEYS = {"id", "text", "author", "work", "citation", "translator", "translation_year",
        "source_url", "source_file", "tags", "status", "replaces"}


def normalise(text: str) -> str:
    """Ignore what typesetting and OCR change: line breaks, hyphenation, quote and dash styles."""
    text = re.sub("[\u200b\u200c\u200d\ufeff\u00ad]", "", text)  # zero-width spaces, soft hyphens
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"-\s+", "-", text)  # "under- stand" split across lines
    text = re.sub(r"\s+([,;:?!.])", r"\1", text)  # OCR puts spaces before punctuation
    return " ".join(text.split())


def test_every_entry_is_complete_and_distinct():
    assert 30 <= len(QUOTES) <= 40
    assert len({q["id"] for q in QUOTES}) == len(QUOTES)
    assert len({normalise(q["text"]).lower() for q in QUOTES}) == len(QUOTES)
    for q in QUOTES:
        assert set(q) == KEYS, q["id"]
        assert q["author"] in {"Marcus Aurelius", "Seneca", "Epictetus"}
        assert q["status"] in {"kept", "reworded", "new"}
        assert q["citation"] and q["translator"] and q["source_url"].startswith("https://")
        assert 1800 < q["translation_year"] < 1930  # public domain


def test_every_mood_the_app_asks_for_has_quotes():
    tags = " ".join(q["tags"] for q in QUOTES).split()
    for mood in ["market_dip", "greed", "euphoria", "uncertainty", "volatility", "patience",
                 "panic", "total_loss", "DCA", "profit_taking", "calm", "neutral", "FOMO",
                 "macro", "recovery"]:
        assert tags.count(mood) >= 1, mood


@pytest.mark.parametrize("quote", QUOTES, ids=[q["id"] for q in QUOTES])
def test_quote_appears_word_for_word_in_its_source(quote):
    path = SOURCES / quote["source_file"]
    if not path.exists():
        pytest.skip("source texts not downloaded (python scripts/download_quote_sources.py)")
    source = normalise(path.read_text(encoding="utf-8", errors="replace"))
    for fragment in normalise(quote["text"]).split("..."):
        fragment = fragment.strip()
        if fragment:
            assert fragment in source, f"{quote['id']}: {fragment!r} not in {quote['source_file']}"
