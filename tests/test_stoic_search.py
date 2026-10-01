"""Quote search: results are always real quotes, and an old index can never serve removed ones."""

import pytest

import stoic_search
from stoic_search import STOIC_CORPUS, StoicSearch, attribution


@pytest.fixture
def keyword_search(monkeypatch):
    monkeypatch.setattr(StoicSearch, "_init_chroma", lambda self, persist_dir: None)
    return StoicSearch()


def test_keyword_search_returns_real_quotes(keyword_search):
    assert keyword_search.mode == "keyword"
    results = keyword_search.query("FOMO crowd noise", n_results=3)
    assert len(results) == 3
    ids = {q["id"] for q in STOIC_CORPUS}
    assert all(r["id"] in ids for r in results)
    assert "FOMO" in results[0]["tags"]


@pytest.mark.parametrize("change", [-0.20, -0.07, 0.0, 0.04, 0.15])
def test_every_market_mood_gets_a_quote(keyword_search, change):
    quote = keyword_search.context_for_change(change)
    assert quote["text"] and quote["citation"]


def test_attribution_names_the_translation():
    q = next(q for q in STOIC_CORPUS if q["id"] == "ma_007")
    assert attribution(q) == "Marcus Aurelius, Meditations 8.47 (tr. George Long, 1862)"


def test_semantic_index_replaces_the_old_unsourced_collection(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path))
    old = client.get_or_create_collection("stoic_wisdom")
    old.add(ids=["sen_013"], documents=["Call no man happy before his death."], embeddings=[[0.1] * 384])
    search = StoicSearch(persist_dir=str(tmp_path))
    if search.mode != "semantic":
        pytest.skip("ChromaDB's embedding model is not available offline here")
    names = {getattr(c, "name", c) for c in chromadb.PersistentClient(path=str(tmp_path)).list_collections()}
    assert names == {stoic_search.COLLECTION_PREFIX + stoic_search.fingerprint(STOIC_CORPUS)}
    best = search.query("I want to sell everything because the price crashed", n_results=3)
    assert len(best) == 3 and all(r["id"] != "sen_013" for r in best)
