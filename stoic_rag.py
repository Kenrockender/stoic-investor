"""
stoic_rag.py
Retrieval-Augmented Generation engine backed by ChromaDB.
Stores curated quotes from Marcus Aurelius, Seneca, and Epictetus,
then retrieves the most relevant one based on market context.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus — 40 hand-picked quotes tagged by emotional context
# ---------------------------------------------------------------------------
STOIC_CORPUS = [
    # --- Market dip / loss ---
    {
        "id": "ma_001",
        "text": "You have power over your mind, not outside events. Realize this, and you will find strength.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "market_dip loss fear anxiety",
    },
    {
        "id": "sen_001",
        "text": "It is not that I'm so brave, but that those who make a mistake punish themselves enough.",
        "author": "Seneca",
        "source": "Letters",
        "tags": "loss regret mistake",
    },
    {
        "id": "ep_001",
        "text": "Seek not that the things which happen should happen as you wish; but wish the things which happen to be as they are, and you will have a tranquil flow of life.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "market_dip acceptance loss volatility",
    },
    {
        "id": "ma_002",
        "text": "The impediment to action advances action. What stands in the way becomes the way.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "market_dip obstacle challenge",
    },
    {
        "id": "sen_002",
        "text": "Omnia, Lucili, aliena sunt, tempus tantum nostrum est. — Everything, Lucilius, belongs to others; time alone is ours.",
        "author": "Seneca",
        "source": "Letters to Lucilius, I",
        "tags": "time patience long_term",
    },
    {
        "id": "ep_002",
        "text": "Make the best use of what is in your power, and take the rest as it happens.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "control market_dip anxiety",
    },
    # --- Greed / euphoria / all-time-high ---
    {
        "id": "sen_003",
        "text": "It is not the man who has too little, but the man who craves more, that is poor.",
        "author": "Seneca",
        "source": "Letters to Lucilius, II",
        "tags": "greed euphoria all_time_high wealth",
    },
    {
        "id": "ma_003",
        "text": "Very little is needed to make a happy life; it is all within yourself, in your way of thinking.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "greed wealth contentment",
    },
    {
        "id": "ep_003",
        "text": "He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.",
        "author": "Epictetus",
        "source": "Fragments",
        "tags": "greed contentment wealth",
    },
    # --- Uncertainty / unknown future ---
    {
        "id": "sen_004",
        "text": "Dum differtur vita transcurrit. — While we are postponing, life speeds by.",
        "author": "Seneca",
        "source": "Letters to Lucilius, I",
        "tags": "uncertainty time opportunity",
    },
    {
        "id": "ma_004",
        "text": "Never let the future disturb you. You will meet it, if you have to, with the same weapons of reason which today arm you against the present.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "uncertainty forecast future",
    },
    {
        "id": "ep_004",
        "text": "I laugh at those who think they can damage me. They do not know who I am, they do not know what I think, they cannot even touch the things which are really mine.",
        "author": "Epictetus",
        "source": "Discourses",
        "tags": "uncertainty fear external_events",
    },
    # --- Volatility / price swings ---
    {
        "id": "ma_005",
        "text": "Loss is nothing else but change, and change is Nature's delight.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "volatility loss change",
    },
    {
        "id": "sen_005",
        "text": "Per aspera ad astra. — Through hardship to the stars.",
        "author": "Seneca",
        "source": "Hercules Furens",
        "tags": "volatility hardship resilience",
    },
    {
        "id": "ep_005",
        "text": "Wealth consists not in having great possessions, but in having few wants.",
        "author": "Epictetus",
        "source": "Fragments",
        "tags": "volatility wealth contentment",
    },
    # --- Patience / long-term holding ---
    {
        "id": "ma_006",
        "text": "Confine yourself to the present.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "patience present_moment long_term",
    },
    {
        "id": "sen_006",
        "text": "Nusquam est qui ubique est. — One who is everywhere is nowhere.",
        "author": "Seneca",
        "source": "Letters to Lucilius, II",
        "tags": "focus patience discipline",
    },
    {
        "id": "ep_006",
        "text": "First say to yourself what you would be; and then do what you have to do.",
        "author": "Epictetus",
        "source": "Discourses",
        "tags": "patience discipline strategy",
    },
    # --- Panic selling ---
    {
        "id": "ma_007",
        "text": "If you are distressed by anything external, the pain is not due to the thing itself, but to your estimate of it; and this you have the power to revoke at any moment.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "panic selling fear market_dip",
    },
    {
        "id": "sen_007",
        "text": "Recede in te ipse quantum potes. — Withdraw into yourself as much as you can.",
        "author": "Seneca",
        "source": "Letters to Lucilius, VII",
        "tags": "panic selling emotion discipline",
    },
    {
        "id": "ep_007",
        "text": "It's not what happens to you, but how you react to it that matters.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "panic selling reaction emotion",
    },
    # --- Death / total loss ---
    {
        "id": "sen_008",
        "text": "Omnia, Lucili, aliena sunt; tempus tantum nostrum est.",
        "author": "Seneca",
        "source": "Letters to Lucilius, I",
        "tags": "total_loss perspective mortality",
    },
    {
        "id": "ma_008",
        "text": "Think of yourself as dead. You have lived your life. Now take what's left and live it properly.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "total_loss perspective reset",
    },
    # --- Discipline / DCA strategy ---
    {
        "id": "ep_008",
        "text": "No man is free who is not master of himself.",
        "author": "Epictetus",
        "source": "Fragments",
        "tags": "discipline DCA strategy consistency",
    },
    {
        "id": "ma_009",
        "text": "Do not indulge in dreams of what you have not, but reckon up the chief of the blessings you do have.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "discipline gratitude DCA",
    },
    {
        "id": "sen_009",
        "text": "Dum differtur vita transcurrit.",
        "author": "Seneca",
        "source": "Letters to Lucilius",
        "tags": "discipline action DCA",
    },
    # --- Profit taking ---
    {
        "id": "ep_009",
        "text": "Seek not the good in external things; seek it in yourself.",
        "author": "Epictetus",
        "source": "Discourses",
        "tags": "profit_taking greed enough",
    },
    {
        "id": "sen_010",
        "text": "Avaritia omnia vitia habet. — Greed possesses all vices.",
        "author": "Seneca",
        "source": "Letters to Lucilius",
        "tags": "profit_taking greed caution",
    },
    # --- Calm / neutral market ---
    {
        "id": "ma_010",
        "text": "You have within you right now, everything you need to deal with whatever the world can throw at you.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "calm neutral strength",
    },
    {
        "id": "ep_010",
        "text": "Demand not that events should happen as you wish; but wish them to happen as they do happen, and you will go on well.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "calm acceptance neutral",
    },
    {
        "id": "sen_011",
        "text": "Vindica te tibi. — Claim yourself for yourself.",
        "author": "Seneca",
        "source": "Letters to Lucilius, I",
        "tags": "calm self_reliance independence",
    },
    # --- FOMO ---
    {
        "id": "ma_011",
        "text": "The object of life is not to be on the side of the majority, but to escape finding oneself in the ranks of the insane.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "FOMO crowd herd_mentality",
    },
    {
        "id": "ep_011",
        "text": "Preach not to others what they should eat, but eat as becomes you, and be silent.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "FOMO noise independence",
    },
    {
        "id": "sen_012",
        "text": "Recede in te ipse quantum potes; cum his versare qui te meliorem facturi sunt.",
        "author": "Seneca",
        "source": "Letters to Lucilius, VII",
        "tags": "FOMO social_media noise",
    },
    # --- Inflation / macro ---
    {
        "id": "ma_012",
        "text": "Accept the things to which fate binds you, and love the people with whom fate brings you together.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "inflation macro acceptance",
    },
    {
        "id": "sen_013",
        "text": "Nemo ante mortem beatus est. — Call no man happy before his death.",
        "author": "Seneca",
        "source": "Letters to Lucilius",
        "tags": "inflation macro long_term",
    },
    # --- Recovery / bounce ---
    {
        "id": "ep_012",
        "text": "We cannot choose our external circumstances, but we can always choose how we respond to them.",
        "author": "Epictetus",
        "source": "Enchiridion",
        "tags": "recovery bounce resilience",
    },
    {
        "id": "ma_013",
        "text": "Nowhere can man find a quieter or more untroubled retreat than in his own soul.",
        "author": "Marcus Aurelius",
        "source": "Meditations",
        "tags": "recovery calm resilience",
    },
    {
        "id": "sen_014",
        "text": "Per aspera ad astra.",
        "author": "Seneca",
        "source": "Hercules Furens",
        "tags": "recovery bounce resilience",
    },
    {
        "id": "ep_013",
        "text": "Difficulties are things that show a person what they are.",
        "author": "Epictetus",
        "source": "Discourses",
        "tags": "recovery hardship character",
    },
]

# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class StoicRAG:
    """
    Lightweight RAG that uses ChromaDB's default sentence-transformer embeddings.
    Falls back to keyword search if ChromaDB is not available.
    """

    COLLECTION_NAME = "stoic_wisdom"

    def __init__(self, persist_dir: str = ".chromadb"):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._persist_dir = persist_dir
        self._fallback = False
        self._init()

    def _init(self):
        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            # Seed if empty
            if self._collection.count() == 0:
                self._seed()
            logger.info("StoicRAG: ChromaDB ready (%d quotes)", self._collection.count())
        except Exception as exc:
            logger.warning("StoicRAG: ChromaDB unavailable (%s) — using keyword fallback", exc)
            self._fallback = True

    def _seed(self):
        ids = [q["id"] for q in STOIC_CORPUS]
        docs = [q["text"] for q in STOIC_CORPUS]
        metas = [
            {"author": q["author"], "source": q["source"], "tags": q["tags"]}
            for q in STOIC_CORPUS
        ]
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # ------------------------------------------------------------------
    def query(self, situation: str, n_results: int = 1) -> list[dict]:
        """
        Return the n most semantically relevant quotes for a given situation.
        Each result: {"text", "author", "source", "distance"}
        """
        if self._fallback:
            return self._keyword_query(situation, n_results)

        try:
            results = self._collection.query(
                query_texts=[situation], n_results=n_results
            )
            out = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                out.append(
                    {
                        "text": doc,
                        "author": meta["author"],
                        "source": meta["source"],
                        "distance": results["distances"][0][i],
                    }
                )
            return out
        except Exception as exc:
            logger.error("StoicRAG query error: %s", exc)
            return self._keyword_query(situation, n_results)

    def _keyword_query(self, situation: str, n: int) -> list[dict]:
        """Simple keyword overlap fallback."""
        tokens = set(situation.lower().split())
        scored = []
        for q in STOIC_CORPUS:
            tag_tokens = set(q["tags"].lower().replace("_", " ").split())
            score = len(tokens & tag_tokens)
            scored.append((score, q))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, q in scored[:n]:
            out.append(
                {
                    "text": q["text"],
                    "author": q["author"],
                    "source": q["source"],
                    "distance": 0.0,
                }
            )
        return out or [
            {
                "text": STOIC_CORPUS[0]["text"],
                "author": STOIC_CORPUS[0]["author"],
                "source": STOIC_CORPUS[0]["source"],
                "distance": 1.0,
            }
        ]

    # ------------------------------------------------------------------
    # Convenience helpers tied to market events
    # ------------------------------------------------------------------
    def on_market_dip(self, pct_change: float) -> dict:
        """pct_change is negative (e.g. -0.07 for -7%)."""
        severity = "severe crash" if pct_change < -0.15 else "market dip loss fear"
        return self.query(severity)[0]

    def on_euphoria(self, pct_change: float) -> dict:
        return self.query("greed euphoria all time high wealth")[0]

    def on_fomo(self) -> dict:
        return self.query("FOMO crowd herd mentality")[0]

    def on_neutral(self) -> dict:
        return self.query("calm neutral patience discipline")[0]

    def context_for_change(self, pct_change: float) -> dict:
        """Auto-select the right philosophical context."""
        if pct_change < -0.05:
            return self.on_market_dip(pct_change)
        elif pct_change > 0.10:
            return self.on_euphoria(pct_change)
        elif -0.02 <= pct_change <= 0.02:
            return self.on_neutral()
        else:
            return self.query("patience resilience")[0]
