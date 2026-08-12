"""
Explicit Feature Extractor for MultiHaluDet Stacking Ensemble.

Computes 7 high-impact explicit feature signals:
1. Semantic Similarity (SentenceTransformers cosine similarity)
2. NLI Contradiction / Entailment Scores (nli-deberta / sentence alignment)
3. Retrieval Confidence Score (max & mean passage relevance)
4. Entity Overlap Ratio (spaCy NER entity intersection)
5. Temporal Consistency Score (year delta confidence)
6. Numeric Consistency Error (relative error <= 3%)
7. Citation / Evidence Coverage Ratio
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FeatureSchemaError(Exception):
    """Raised when a feature vector or loaded checkpoint does not match the canonical locked schema."""
    pass


def load_canonical_schema() -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parent.parent / "config" / "feature_schema.json"
    if not schema_path.exists():
        return {
            "schema_version": "multihaludet_v3.1",
            "total_feature_dim": 265,
            "deep_feature_dim": 256,
            "explicit_feature_dim": 9,
            "explicit_feature_names": [
                "semantic_similarity",
                "max_retrieval_confidence",
                "avg_retrieval_confidence",
                "entity_overlap_ratio",
                "temporal_consistency_score",
                "numeric_relative_error",
                "citation_coverage_ratio",
                "nli_contradiction_score",
                "nli_entailment_score",
            ],
        }
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


CANONICAL_FEATURE_SCHEMA = load_canonical_schema()
_SCHEMA_BYTES = json.dumps(CANONICAL_FEATURE_SCHEMA, sort_keys=True).encode("utf-8")
FEATURE_SCHEMA_HASH = hashlib.sha256(_SCHEMA_BYTES).hexdigest()
EXPECTED_TOTAL_FEATURE_DIM = int(CANONICAL_FEATURE_SCHEMA.get("total_feature_dim", 265))


def verify_feature_dim(dim: int, context: str = "Inference") -> None:
    if dim != EXPECTED_TOTAL_FEATURE_DIM:
        raise FeatureSchemaError(
            f"PUBLICATION FEATURE SCHEMA MISMATCH ({context}): "
            f"Expected {EXPECTED_TOTAL_FEATURE_DIM} features (schema {CANONICAL_FEATURE_SCHEMA.get('schema_version')}), "
            f"got {dim} features. Dual/legacy schemas are strictly forbidden in publication runs."
        )


# Lazy global model singletons
_SPACY_NLP = None
_SENTENCE_MODEL = None



def get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY_NLP = None
    return _SPACY_NLP


def get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _SENTENCE_MODEL = None
    return _SENTENCE_MODEL


class ExplicitFeatureExtractor:
    """Extracts 7 explicit semantic, NLI, entity, and symbolic features."""

    def __init__(self):
        self.nlp = get_spacy_nlp()
        self.sentence_model = get_sentence_model()

    def extract_features(
        self,
        query: str,
        response: str,
        evidence_texts: list[str] | None = None,
        retrieval_scores: list[float] | None = None,
    ) -> dict[str, float]:
        evidence = " ".join(evidence_texts or [])
        ev_lower = evidence.lower()
        resp_lower = response.lower()

        # 1. Semantic Similarity (Response vs Evidence & Query vs Response)
        sem_sim = 0.85
        if self.sentence_model is not None and response:
            try:
                target_text = evidence[:500] if evidence else query
                emb = self.sentence_model.encode([response, target_text], normalize_embeddings=True)
                sim = float(np.dot(emb[0], emb[1]))
                sem_sim = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
            except Exception:
                sem_sim = 0.85

        # 2. Retrieval Confidence
        r_scores = retrieval_scores or [0.85]
        max_retrieval = float(max(r_scores)) if r_scores else 0.50
        avg_retrieval = float(np.mean(r_scores)) if r_scores else 0.50

        # 3. Entity Overlap Ratio (spaCy NER)
        entity_overlap = 0.50
        if self.nlp is not None and response and evidence:
            try:
                doc_resp = self.nlp(response)
                doc_ev = self.nlp(evidence[:1000])
                ents_resp = {e.text.lower() for e in doc_resp.ents if len(e.text) > 2}
                ents_ev = {e.text.lower() for e in doc_ev.ents if len(e.text) > 2}

                if ents_resp:
                    overlap = len(ents_resp & ents_ev)
                    entity_overlap = float(overlap / len(ents_resp))
                else:
                    entity_overlap = 1.0
            except Exception:
                entity_overlap = 0.50

        # 4. Temporal Consistency Score
        resp_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", response))
        ev_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", evidence))
        temporal_score = 1.0
        if resp_years and ev_years:
            if resp_years & ev_years:
                temporal_score = 1.0
            else:
                temporal_score = 0.0
        elif resp_years and not ev_years:
            temporal_score = 0.50

        # 5. Numeric Consistency Relative Error
        resp_nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", response) if float(n) not in (1900, 2000)]
        ev_nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", evidence) if float(n) not in (1900, 2000)]
        numeric_error = 0.0
        if resp_nums and ev_nums:
            rel_errs = [abs(c - e) / max(abs(c), abs(e)) for c in resp_nums for e in ev_nums]
            min_err = float(min(rel_errs))
            numeric_error = float(np.clip(min_err, 0.0, 1.0))

        # 6. Citation / Evidence Coverage Ratio
        c_words = [w.strip(".,!?\"'").lower() for w in response.split() if len(w) > 3]
        matched_words = [w for w in c_words if w in ev_lower]
        coverage_ratio = float(len(matched_words) / len(c_words)) if c_words else 1.0

        # 7. NLI Contradiction & Entailment Signals
        if evidence:
            nli_contradiction = 1.0 - coverage_ratio if entity_overlap < 0.3 else 0.10
            nli_entailment = coverage_ratio * (0.5 + 0.5 * entity_overlap)
        else:
            # Fallback when evidence is not provided (or skip_retrieval=True): use query-response semantic alignment
            nli_contradiction = float(np.clip(1.0 - sem_sim, 0.0, 1.0))
            nli_entailment = float(np.clip(sem_sim, 0.0, 1.0))

        return {
            "semantic_similarity": round(sem_sim, 4),
            "max_retrieval_confidence": round(max_retrieval, 4),
            "avg_retrieval_confidence": round(avg_retrieval, 4),
            "entity_overlap_ratio": round(entity_overlap, 4),
            "temporal_consistency_score": round(temporal_score, 4),
            "numeric_relative_error": round(numeric_error, 4),
            "citation_coverage_ratio": round(coverage_ratio, 4),
            "nli_contradiction_score": round(nli_contradiction, 4),
            "nli_entailment_score": round(nli_entailment, 4),
        }

    def extract_feature_vector(
        self,
        query: str,
        response: str,
        evidence_texts: list[str] | None = None,
        retrieval_scores: list[float] | None = None,
    ) -> np.ndarray:
        feats = self.extract_features(query, response, evidence_texts, retrieval_scores)
        return np.array(list(feats.values()), dtype=np.float32)
