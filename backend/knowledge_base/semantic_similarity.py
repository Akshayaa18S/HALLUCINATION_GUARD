"""
Optional semantic similarity scorer for evidence ranking (Phase 7/12).

Uses sentence-transformers + FAISS if both are installed. If not, ranker.py
falls back to lexical word-overlap similarity, which is weaker but requires
no extra downloads - keeps the pipeline runnable out of the box.
"""

import logging

logger = logging.getLogger(__name__)

_model = None
_faiss = None
_load_attempted = False


def _ensure_loaded():
    global _model, _faiss, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _faiss = faiss
        logger.info("Semantic similarity: using sentence-transformers + FAISS")
    except Exception as exc:
        logger.warning(
            "Semantic similarity backend unavailable (%s) - evidence ranking "
            "will use lexical overlap instead. For better ranking: "
            "pip install sentence-transformers faiss-cpu",
            exc,
        )
        _model = None
        _faiss = None


def is_available() -> bool:
    _ensure_loaded()
    return _model is not None


def similarity(query: str, candidates: list[str]) -> list[float]:
    """Returns cosine-similarity-like scores in [0, 1] for each candidate
    against the query. Caller (ranker.py) only uses this when
    is_available() is True."""
    _ensure_loaded()
    if _model is None or not candidates:
        return [0.0] * len(candidates)

    import numpy as np

    query_emb = _model.encode([query], normalize_embeddings=True)
    cand_emb = _model.encode(candidates, normalize_embeddings=True)

    index = _faiss.IndexFlatIP(query_emb.shape[1])
    index.add(cand_emb.astype("float32"))
    scores, _ = index.search(query_emb.astype("float32"), len(candidates))
    # scores are for the single query row, in the order faiss ranked them -
    # but we want them aligned to `candidates` order, so search per-item instead.
    return [float(np.dot(query_emb[0], c)) for c in cand_emb]
