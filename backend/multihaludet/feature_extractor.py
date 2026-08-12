"""
Explicit Feature Extractor for MultiHaluDet Stacking Ensemble (Schema v3.2).

Computes 15 high-impact explicit feature signals:
1. Semantic Similarity (SentenceTransformers cosine similarity)
2. Retrieval Confidence Score (max & mean passage relevance)
3. Entity Overlap Ratio (spaCy NER entity intersection)
4. Localized Temporal Consistency Score (sentence-matched year delta confidence)
5. Localized Numeric Consistency Relative Error (claim-to-sentence relative error)
6. Citation / Evidence Coverage Ratio
7. Genuine NLI Contradiction Score (max claim-level contradiction probability)
8. Genuine NLI Entailment Score (mean claim-level entailment probability)
9. Genuine NLI Neutral Score (mean claim-level neutral probability)
10. Semantic Available (explicit missingness flag)
11. Entity Available (explicit missingness flag)
12. Evidence Available (explicit missingness flag)
13. NLI Available (explicit missingness flag)
14. Numeric Available (explicit missingness flag)
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
            "schema_version": "multihaludet_v3.2",
            "total_feature_dim": 271,
            "deep_feature_dim": 256,
            "explicit_feature_dim": 15,
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
                "nli_neutral_score",
                "semantic_available",
                "entity_available",
                "evidence_available",
                "nli_available",
                "numeric_available",
            ],
        }
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


CANONICAL_FEATURE_SCHEMA = load_canonical_schema()
_SCHEMA_BYTES = json.dumps(CANONICAL_FEATURE_SCHEMA, sort_keys=True).encode("utf-8")
FEATURE_SCHEMA_HASH = hashlib.sha256(_SCHEMA_BYTES).hexdigest()
EXPECTED_TOTAL_FEATURE_DIM = int(CANONICAL_FEATURE_SCHEMA.get("total_feature_dim", 271))
EXPLICIT_FEATURE_NAMES = list(CANONICAL_FEATURE_SCHEMA.get("explicit_feature_names", []))



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
_NLI_PIPELINE = None


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


def get_nli_pipeline(device: str | int | None = None, strict_nli: bool = False):
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        try:
            import torch
            from transformers import pipeline
            dev_target = device
            if dev_target is None:
                dev_target = 0 if torch.cuda.is_available() else -1
            pipe_kwargs = {"model": "cross-encoder/nli-deberta-v3-base", "top_k": None, "device": dev_target}
            _NLI_PIPELINE = pipeline("text-classification", **pipe_kwargs)
            dev_label = f"cuda:{dev_target}" if (isinstance(dev_target, int) and dev_target >= 0) or str(dev_target).startswith("cuda") else "cpu"
            logger.info("NLI device: %s | NLI model: cross-encoder/nli-deberta-v3-base", dev_label)
        except Exception as exc:
            if strict_nli:
                raise RuntimeError(
                    f"ERROR: DeBERTa NLI model 'cross-encoder/nli-deberta-v3-base' unavailable under strict_nli=True ({exc}). "
                    "Silent fallback is disabled for publication runs. Pass --allow-nli-fallback to enable non-publication fallback."
                ) from exc
            _NLI_PIPELINE = None
    return _NLI_PIPELINE


def split_sentences(text: str) -> list[str]:
    """Splits text into clean sentences/claims."""
    if not text or not text.strip():
        return []
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 3]


class ExplicitFeatureExtractor:
    """Extracts explicit semantic, NLI, localized numeric, entity, and missingness features."""

    def __init__(self, device: str | int | None = None, strict_nli: bool = False):
        self.strict_nli = strict_nli
        self.device = device
        self.nlp = get_spacy_nlp()
        self.sentence_model = get_sentence_model()
        try:
            self.nli_pipeline = get_nli_pipeline(device=device, strict_nli=strict_nli)
        except Exception:
            if strict_nli:
                raise
            self.nli_pipeline = None



    def extract_features(
        self,
        query: str,
        response: str,
        evidence_texts: list[str] | None = None,
        retrieval_scores: list[float] | None = None,
    ) -> dict[str, float]:
        if not evidence_texts and query:
            evidence_texts = [query]
        evidence = " ".join(evidence_texts or []).strip()
        ev_lower = evidence.lower()
        resp_lower = response.lower()

        # Missingness indicator flags
        semantic_available = 0.0
        entity_available = 0.0
        evidence_available = 1.0 if evidence else 0.0
        nli_available = 1.0 if self.nli_pipeline is not None else 0.0
        numeric_available = 0.0


        # 1. Semantic Similarity (Neutral default: 0.50 if unavailable, NOT 0.85)
        sem_sim = 0.50
        if self.sentence_model is not None and response:
            try:
                target_text = evidence[:500] if evidence else query
                emb = self.sentence_model.encode([response, target_text], normalize_embeddings=True)
                sim = float(np.dot(emb[0], emb[1]))
                sem_sim = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
                semantic_available = 1.0
            except Exception:
                sem_sim = 0.50
                semantic_available = 0.0

        # 2. Retrieval Confidence
        r_scores = retrieval_scores or []
        if r_scores:
            max_retrieval = float(max(r_scores))
            avg_retrieval = float(np.mean(r_scores))
        elif evidence:
            max_retrieval = 0.50
            avg_retrieval = 0.50
        else:
            max_retrieval = 0.0
            avg_retrieval = 0.0

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
                    entity_available = 1.0
                else:
                    entity_overlap = 0.50
                    entity_available = 0.0
            except Exception:
                entity_overlap = 0.50
                entity_available = 0.0

        # 4. Localized Sentence Matching for Numeric & Temporal Features
        resp_sentences = split_sentences(response)
        ev_sentences = split_sentences(evidence) if evidence else []

        # Localized Temporal Score
        resp_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", response))
        ev_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", evidence))
        temporal_score = 0.50
        if resp_years and ev_years:
            matched_years = sum(1 for y in resp_years if y in ev_years)
            temporal_score = float(matched_years / len(resp_years))
        elif resp_years and not ev_years:
            temporal_score = 0.0

        # Localized Numeric Consistency Error
        # Local claim matching: compare numbers per claim sentence to matched evidence sentence containing numbers
        numeric_errors: list[float] = []
        for s_resp in resp_sentences:
            s_nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", s_resp) if float(n) not in (1900, 2000)]
            if not s_nums:
                continue

            # Find evidence sentences that contain numeric entities
            ev_s_with_nums = [
                (s_ev, [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", s_ev) if float(n) not in (1900, 2000)])
                for s_ev in ev_sentences
            ]
            ev_s_with_nums = [(s_ev, nums) for s_ev, nums in ev_s_with_nums if nums]

            if ev_s_with_nums:
                # Pick the evidence sentence with numbers that has highest word overlap/relevance to s_resp
                best_ev_s, ev_s_nums = max(
                    ev_s_with_nums,
                    key=lambda pair: len(set(s_resp.lower().split()) & set(pair[0].lower().split()))
                )
                numeric_available = 1.0
                for c_num in s_nums:
                    # Find relative error to closest number in matched evidence sentence
                    errs = [abs(c_num - e_num) / max(abs(c_num), abs(e_num), 1e-5) for e_num in ev_s_nums]
                    numeric_errors.append(float(min(errs)))
            else:
                if evidence:
                    # Number in claim sentence has no corresponding number in matched evidence
                    numeric_errors.append(1.0)
                    numeric_available = 1.0

        numeric_error = float(np.mean(numeric_errors)) if numeric_errors else 0.0


        # 6. Citation / Evidence Coverage Ratio
        c_words = [w.strip(".,!?\"'").lower() for w in response.split() if len(w) > 3]
        matched_words = [w for w in c_words if w in ev_lower] if ev_lower else []
        coverage_ratio = float(len(matched_words) / len(c_words)) if c_words else 0.0

        # 7. Genuine Claim-Level NLI Model Signals
        nli_contradiction = 0.0
        nli_entailment = 0.50
        nli_neutral = 0.50

        if evidence and resp_sentences:
            if self.nli_pipeline is not None:
                try:
                    c_probs, e_probs, n_probs = [], [], []
                    # Resolve id2label mapping dynamically from model config if available
                    id2label_map = {}
                    if hasattr(self.nli_pipeline, "model") and hasattr(self.nli_pipeline.model, "config"):
                        raw_id2label = getattr(self.nli_pipeline.model.config, "id2label", None) or {}
                        for k, v in raw_id2label.items():
                            id2label_map[str(k).lower()] = str(v).lower()
                            id2label_map[f"label_{k}".lower()] = str(v).lower()

                    num_labels = getattr(getattr(self.nli_pipeline, "model", None), "config", None)
                    top_k_val = getattr(num_labels, "num_labels", 3) or 3

                    for claim in resp_sentences[:5]:  # Process top 5 claims
                        res = self.nli_pipeline({"text": evidence[:1000], "text_pair": claim}, top_k=top_k_val)
                        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                            res = res[0]
                        score_dict = {}
                        for item in res:
                            lbl = str(item["label"]).lower()
                            val = float(item["score"])
                            target = id2label_map.get(lbl, lbl)
                            if "contradiction" in target or target == "label_0":
                                score_dict["contradiction"] = val
                            elif "entailment" in target or target == "label_1":
                                score_dict["entailment"] = val
                            elif "neutral" in target or target == "label_2":
                                score_dict["neutral"] = val

                        c_probs.append(score_dict.get("contradiction", 0.0))
                        e_probs.append(score_dict.get("entailment", 0.0))
                        n_probs.append(score_dict.get("neutral", 0.0))


                    nli_contradiction = float(max(c_probs)) if c_probs else 0.0
                    nli_entailment = float(np.mean(e_probs)) if e_probs else 0.50
                    nli_neutral = float(np.mean(n_probs)) if n_probs else 0.50
                    nli_available = 1.0
                except Exception as exc:
                    if self.strict_nli:
                        raise RuntimeError(
                            f"STRICT NLI FAIL-CLOSED ERROR: DeBERTa NLI inference failed under strict_nli=True ({exc}). "
                            "Silent fallback to pseudo-NLI features is strictly forbidden in publication runs."
                        ) from exc
                    logger.debug("NLI DeBERTa pipeline error: %s", exc)

            if nli_available == 0.0 and not self.strict_nli and self.sentence_model is not None and ev_sentences:
                try:
                    # Embedding alignment fallback for claim-level NLI (Development Mode ONLY)
                    c_scores = []
                    for claim in resp_sentences[:5]:
                        emb_c = self.sentence_model.encode([claim], normalize_embeddings=True)[0]
                        emb_evs = self.sentence_model.encode(ev_sentences[:10], normalize_embeddings=True)
                        sims = np.dot(emb_evs, emb_c)
                        max_sim = float(max(sims))
                        c_scores.append(max_sim)
                    # Convert similarities to probability distribution
                    avg_c_sim = float(np.mean(c_scores)) if c_scores else 0.5
                    nli_entailment = float(np.clip(avg_c_sim, 0.0, 1.0))
                    nli_contradiction = float(np.clip(1.0 - avg_c_sim, 0.0, 1.0))
                    nli_neutral = float(np.clip(1.0 - abs(avg_c_sim - 0.5) * 2, 0.0, 1.0))
                    nli_available = 1.0
                except Exception:
                    pass

        if nli_available == 0.0:
            if self.strict_nli:
                raise RuntimeError(
                    "STRICT NLI FAIL-CLOSED ERROR: NLI features are unavailable under strict_nli=True. "
                    "Pseudo-NLI fallbacks (1.0 - sem_sim) are strictly disabled for publication runs."
                )
            # Fallback when no model/evidence (Development Mode ONLY): use query semantic alignment
            nli_contradiction = float(np.clip(1.0 - sem_sim, 0.0, 1.0))
            nli_entailment = float(np.clip(sem_sim, 0.0, 1.0))
            nli_neutral = float(np.clip(1.0 - abs(sem_sim - 0.5) * 2, 0.0, 1.0))


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
            "nli_neutral_score": round(nli_neutral, 4),
            "semantic_available": float(semantic_available),
            "entity_available": float(entity_available),
            "evidence_available": float(evidence_available),
            "nli_available": float(nli_available),
            "numeric_available": float(numeric_available),
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

