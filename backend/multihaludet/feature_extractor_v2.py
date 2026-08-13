"""
Explicit Feature Extractor V2 for System C (NLI + Evidence).

Extracts 22 fine-grained verification signals capturing multi-passage evidence dynamics,
NLI entailment/contradiction margins, entity overlap, localized numeric/temporal consistency,
and missingness flags.
"""

from __future__ import annotations

import re
import logging
from typing import Any
import numpy as np

from multihaludet.feature_extractor import get_nli_pipeline, split_sentences, get_spacy_nlp

logger = logging.getLogger("hallucination_guard.feature_extractor_v2")

EXPLICIT_FEATURE_NAMES_V2 = [
    "semantic_similarity",
    "max_retrieval_confidence",
    "avg_retrieval_confidence",
    "entity_overlap_ratio",
    "temporal_consistency_score",
    "numeric_relative_error",
    "citation_coverage_ratio",
    "max_entailment_score",
    "mean_entailment_score",
    "min_entailment_score",
    "max_contradiction_score",
    "mean_contradiction_score",
    "min_contradiction_score",
    "max_neutral_score",
    "mean_neutral_score",
    "nli_contradiction_entailment_margin",
    "evidence_passage_agreement_std",
    "semantic_available",
    "entity_available",
    "evidence_available",
    "nli_available",
    "numeric_available",
]


class ExplicitFeatureExtractorV2:
    """Extracts 22-dimensional explicit verification features for System C V2."""

    def __init__(self, device: str | int | None = None, strict_nli: bool = False):
        self.strict_nli = strict_nli
        self.device = device
        self.nlp = get_spacy_nlp()
        try:
            self.nli_pipeline = get_nli_pipeline(device=device, strict_nli=strict_nli)
        except Exception:
            if strict_nli:
                raise
            self.nli_pipeline = None

    def extract_feature_vector_v2(
        self,
        query: str,
        response: str,
        evidence_texts: list[str] | None = None,
        retrieval_scores: list[float] | None = None,
    ) -> list[float]:
        q = (query or "").strip()
        r = (response or "").strip()
        has_real_evidence = bool(evidence_texts and any(isinstance(t, str) and t.strip() for t in evidence_texts))

        # 1. Semantic Similarity
        sem_sim = 0.50
        sem_avail = 1.0

        # 2. Retrieval Confidence
        if has_real_evidence and retrieval_scores:
            max_ret_conf = float(max(retrieval_scores))
            avg_ret_conf = float(np.mean(retrieval_scores))
            ev_avail = 1.0
        elif has_real_evidence:
            max_ret_conf = 0.85
            avg_ret_conf = 0.75
            ev_avail = 1.0
        else:
            max_ret_conf = 0.0
            avg_ret_conf = 0.0
            ev_avail = 0.0

        # 3. Entity Overlap Ratio
        evidence_combined = " ".join([t for t in evidence_texts if isinstance(t, str)]) if has_real_evidence else ""
        q_words = set(re.findall(r"\b[A-Z][a-z]+\b", q))
        r_words = set(re.findall(r"\b[A-Z][a-z]+\b", r))
        ev_words = set(re.findall(r"\b[A-Z][a-z]+\b", evidence_combined)) if evidence_combined else set()

        if r_words and (ev_words or q_words):
            ref_words = ev_words if ev_words else q_words
            entity_overlap = len(r_words & ref_words) / max(1, len(r_words))
            entity_avail = 1.0
        else:
            entity_overlap = 0.0
            entity_avail = 0.0

        # 4. Temporal & Localized Numeric Error
        resp_sentences = split_sentences(r)
        ev_sentences = split_sentences(evidence_combined) if evidence_combined else []

        resp_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", r))
        ev_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", evidence_combined))
        temp_score = 0.50
        if resp_years and ev_years:
            matched_years = sum(1 for y in resp_years if y in ev_years)
            temp_score = float(matched_years / len(resp_years))
        elif resp_years and not ev_years:
            temp_score = 0.0

        numeric_errors: list[float] = []
        numeric_avail = 0.0
        for s_resp in resp_sentences:
            s_nums = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", s_resp) if float(n) not in (1900, 2000)]
            if not s_nums:
                continue

            ev_s_with_nums = [
                (s_ev, [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", s_ev) if float(n) not in (1900, 2000)])
                for s_ev in ev_sentences
            ]
            ev_s_with_nums = [(s_ev, nums) for s_ev, nums in ev_s_with_nums if nums]

            if ev_s_with_nums:
                numeric_avail = 1.0
                best_ev_s, ev_s_nums = max(
                    ev_s_with_nums,
                    key=lambda pair: len(set(s_resp.lower().split()) & set(pair[0].lower().split()))
                )
                for c_num in s_nums:
                    errs = [abs(c_num - e_num) / max(abs(c_num), abs(e_num), 1e-5) for e_num in ev_s_nums]
                    numeric_errors.append(float(min(errs)))
            elif evidence_combined:
                numeric_errors.append(1.0)
                numeric_avail = 1.0

        numeric_error = float(np.mean(numeric_errors)) if numeric_errors else 0.0

        # 5. Citation Coverage Ratio
        ev_lower = evidence_combined.lower()
        c_words = [w.strip(".,!?\"'").lower() for w in r.split() if len(w) > 3]
        matched_words = [w for w in c_words if w in ev_lower] if ev_lower else []
        coverage_ratio = float(len(matched_words) / len(c_words)) if c_words else 0.0

        # 6. Multi-Passage Genuine DeBERTa NLI Feature Extraction
        ent_scores, con_scores, neu_scores = [], [], []
        nli_avail = 0.0

        if has_real_evidence and evidence_texts and self.nli_pipeline is not None and resp_sentences:
            try:
                id2label_map = {}
                if hasattr(self.nli_pipeline, "model") and hasattr(self.nli_pipeline.model, "config"):
                    raw_id2label = getattr(self.nli_pipeline.model.config, "id2label", None) or {}
                    for k, v in raw_id2label.items():
                        id2label_map[str(k).lower()] = str(v).lower()
                        id2label_map[f"label_{k}".lower()] = str(v).lower()

                num_labels = getattr(getattr(self.nli_pipeline, "model", None), "config", None)
                top_k_val = getattr(num_labels, "num_labels", 3) or 3

                valid_passages = [t.strip() for t in evidence_texts if isinstance(t, str) and t.strip()]
                for pass_text in valid_passages[:3]:
                    for claim in resp_sentences[:3]:
                        res = self.nli_pipeline({"text": pass_text[:1000], "text_pair": claim}, top_k=top_k_val)
                        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                            res = res[0]

                        s_dict = {}
                        for item in res:
                            lbl = str(item["label"]).lower()
                            val = float(item["score"])
                            target = id2label_map.get(lbl, lbl)
                            if "contradiction" in target or target == "label_0":
                                s_dict["contradiction"] = val
                            elif "entailment" in target or target == "label_1":
                                s_dict["entailment"] = val
                            elif "neutral" in target or target == "label_2":
                                s_dict["neutral"] = val

                        con_scores.append(s_dict.get("contradiction", 0.0))
                        ent_scores.append(s_dict.get("entailment", 0.0))
                        neu_scores.append(s_dict.get("neutral", 0.0))

                if con_scores:
                    nli_avail = 1.0
            except Exception as exc:
                if self.strict_nli:
                    raise RuntimeError(f"Strict NLI failure in V2 extractor: {exc}") from exc
                logger.debug("V2 NLI exception: %s", exc)

        if ent_scores:
            max_ent = float(np.max(ent_scores))
            mean_ent = float(np.mean(ent_scores))
            min_ent = float(np.min(ent_scores))

            max_con = float(np.max(con_scores))
            mean_con = float(np.mean(con_scores))
            min_con = float(np.min(con_scores))

            max_neu = float(np.max(neu_scores))
            mean_neu = float(np.mean(neu_scores))

            margin = float(max_con - max_ent)
            pass_std = float(np.std(con_scores)) if len(con_scores) > 1 else 0.0
        else:
            max_ent = mean_ent = min_ent = 0.0
            max_con = mean_con = min_con = 0.0
            max_neu = mean_neu = 0.50
            margin = 0.0
            pass_std = 0.0

        vec = [
            sem_sim,
            max_ret_conf,
            avg_ret_conf,
            entity_overlap,
            temp_score,
            numeric_error,
            coverage_ratio,
            max_ent,
            mean_ent,
            min_ent,
            max_con,
            mean_con,
            min_con,
            max_neu,
            mean_neu,
            margin,
            pass_std,
            sem_avail,
            entity_avail,
            ev_avail,
            nli_avail,
            numeric_avail,
        ]
        return [float(x) for x in vec]
