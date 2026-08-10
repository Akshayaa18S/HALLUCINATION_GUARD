"""Semantic Evidence Sentence Selector module for Hallucination Guard.

Extracts, cleans, ranks, and selects top supporting sentences from retrieved document text
based on semantic similarity, keyword overlap, named entity overlap, and claim coverage.
"""

from dataclasses import dataclass
import re
import time
from typing import Any


from config.settings import settings


@dataclass
class SentenceScore:
    sentence: str
    score: float
    support_score: float
    contradiction_score: float
    neutral_score: float
    sim_score: float
    ent_score: float
    cov_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "score": round(self.score, 4),
            "support_score": round(self.support_score, 4),
            "contradiction_score": round(self.contradiction_score, 4),
            "neutral_score": round(self.neutral_score, 4),
            "sim_score": round(self.sim_score, 4),
            "ent_score": round(self.ent_score, 4),
            "cov_score": round(self.cov_score, 4),
        }


def np_clip_val(val: float, min_val: float, max_val: float) -> float:
    return min(max_val, max(min_val, val))


class EvidenceSelector:
    """Ranks and selects top supporting sentences from retrieved document text."""

    _CITATION_REGEX = re.compile(r"\[\d+\]|\[citation needed\]|\[edit\]", re.IGNORECASE)
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "to", "in", "it",
        "by", "with", "from", "at", "on", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "also", "not", "only"
    }

    @classmethod
    def clean_text_into_sentences(cls, raw_text: str) -> list[str]:
        """Clean Wikipedia citations and split document into non-empty, verb-containing sentences while preserving order."""
        if not raw_text:
            return []

        # Remove Wikipedia citations like [1], [citation needed]
        cleaned_text = cls._CITATION_REGEX.sub("", raw_text)

        # Split into sentences using sentence-ending punctuation (.!?)
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned_text.strip())

        sentences = []
        _ACTION_VERBS = {"is", "was", "are", "were", "formed", "founded", "debuted", "released", "became", "served", "located", "headquartered", "born", "produced", "created", "directed", "governs", "known"}

        for s in raw_sentences:
            s_clean = s.strip()
            # Filter out empty or excessively short fragments, headers (==), or title fragments lacking verbs
            if len(s_clean) >= 20 and not s_clean.startswith("==") and not s_clean.startswith("—"):
                s_words = set(w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", s_clean))
                # Must contain at least one verb/copula to be a valid factual assertion sentence (filtering out paper/book titles)
                if s_words.intersection(_ACTION_VERBS) or len(s_words) >= 6:
                    sentences.append(s_clean)

        return sentences if sentences else [s.strip() for s in raw_sentences if len(s.strip()) >= 15]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\b", text.lower())
        return [w for w in words if w not in cls._STOPWORDS and len(w) > 1]

    @classmethod
    def compute_sentence_score(
        cls,
        sentence: str,
        claim: str,
        target_entities: list[str] | None = None,
        sentence_idx: int = 0,
    ) -> SentenceScore:
        """Compute composite ranking score S_rank, support, contradiction, and neutral scores relative to claim.

        Formula:
        S_rank = 0.40 * S_entity + 0.30 * S_relation + 0.20 * S_density + 0.10 * S_lead_bonus
        """
        s_tokens = cls._tokenize(sentence)
        c_tokens = cls._tokenize(claim)

        if not s_tokens or not c_tokens:
            return SentenceScore(
                sentence=sentence,
                score=0.10,
                support_score=0.10,
                contradiction_score=0.0,
                neutral_score=0.90,
                sim_score=0.10,
                ent_score=0.0,
                cov_score=0.0,
            )

        set_s = set(s_tokens)
        set_c = set(c_tokens)
        s_norm = sentence.lower()
        c_norm = claim.lower()

        # 1. Semantic Similarity Proxy (Jaccard + Overlap Coefficient)
        intersection = set_s.intersection(set_c)
        jaccard = len(intersection) / len(set_s.union(set_c))
        overlap_coef = len(intersection) / min(len(set_s), len(set_c))
        sim_score = float(0.5 * jaccard + 0.5 * overlap_coef)

        # 2. Entity Overlap & Proper Noun Alignment
        proper_nouns = set(w for w in re.findall(r"\b[A-Z][a-z0-9]+\b", claim) if w.lower() not in cls._STOPWORDS)
        if target_entities:
            for te in target_entities:
                for w in te.split():
                    if w[0].isupper() and w.lower() not in cls._STOPWORDS:
                        proper_nouns.add(w)

        if proper_nouns:
            ent_matches = sum(1 for pn in proper_nouns if pn.lower() in s_norm)
            ent_score = float(ent_matches / len(proper_nouns))
        else:
            ent_score = sim_score

        # 3. Relational Entailment & Head Entity Co-occurrence
        head_pn_claim = set(pn.lower() for pn in proper_nouns if len(pn) > 2)
        s_pn_count = sum(1 for head in head_pn_claim if head in s_norm)
        rel_entail_score = float(s_pn_count / max(1, len(head_pn_claim))) if head_pn_claim else sim_score

        # 4. Factual Density (ratio of non-stopword tokens and proper nouns/verbs)
        factual_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", sentence) if w.lower() not in cls._STOPWORDS]
        factual_density = float(min(1.0, len(factual_tokens) / max(1, len(sentence.split()))))

        # 5. Lead Position Bonus (first 3 sentences of Wikipedia definition get lead bonus)
        lead_bonus = 1.0 if sentence_idx < 3 else (0.50 if sentence_idx < 5 else 0.10)

        # 6. Generic Contradiction & Support Detection
        negation_claim = any(neg in c_norm for neg in ("not", "no", "never", "is not", "was not", "didn't"))
        negation_sent = any(neg in s_norm for neg in ("not", "no", "never", "is not", "was not", "didn't"))

        if negation_claim != negation_sent and ent_score >= 0.50:
            contradiction_score = float(min(1.0, 0.70 + 0.30 * ent_score))
            support_score = float(max(0.0, 0.30 * sim_score))
        else:
            contradiction_score = float(max(0.0, 0.10 * (1.0 - sim_score)))
            support_score = float(min(1.0, 0.50 * sim_score + 0.50 * rel_entail_score))

        neutral_score = float(max(0.0, 1.0 - (support_score + contradiction_score)))

        # Composite S_rank = 0.40 ent + 0.30 rel + 0.20 density + 0.10 lead
        s_rank = float(np_clip_val(0.40 * ent_score + 0.30 * rel_entail_score + 0.20 * factual_density + 0.10 * lead_bonus, 0.0, 1.0))

        return SentenceScore(
            sentence=sentence,
            score=s_rank,
            support_score=round(support_score, 4),
            contradiction_score=round(contradiction_score, 4),
            neutral_score=round(neutral_score, 4),
            sim_score=round(sim_score, 4),
            ent_score=round(ent_score, 4),
            cov_score=round(factual_density, 4),
        )

    def select_best_sentences(
        self,
        claim: str,
        document_text: str,
        top_k: int = 3,
        target_entities: list[str] | None = None,
        min_relative_ratio: float = 0.80,
    ) -> dict[str, Any]:
        """Extract and rank sentences from document_text, returning top supporting sentences and evidence strength."""
        start_time = time.monotonic()
        sentences = self.clean_text_into_sentences(document_text)

        if not sentences:
            return {
                "supporting_sentences": [document_text[:250]] if document_text else [],
                "best_excerpt": document_text[:250] if document_text else "",
                "evidence_strength": 0.50,
                "sentence_ranking": [],
                "execution_ms": round((time.monotonic() - start_time) * 1000.0, 2),
            }

        scores: list[SentenceScore] = []
        for idx, s in enumerate(sentences):
            score_obj = self.compute_sentence_score(s, claim, target_entities=target_entities, sentence_idx=idx)
            scores.append(score_obj)

        # Sort sentences descending by score
        ranked_scores = sorted(scores, key=lambda x: x.score, reverse=True)
        top_score_val = ranked_scores[0].score if ranked_scores else 0.50

        # Adaptive threshold filtering: Retain only sentences scoring >= 80% of top score
        threshold_score = top_score_val * min_relative_ratio
        filtered_objs = [obj for obj in ranked_scores if obj.score >= threshold_score and obj.score >= 0.35]
        if not filtered_objs:
            filtered_objs = [ranked_scores[0]]

        top_sentences_objs = filtered_objs[:top_k]
        top_sentences = [obj.sentence for obj in top_sentences_objs]
        best_excerpt = top_sentences[0] if top_sentences else ""

        # Computed Evidence Strength: 0.45 sim + 0.30 ent + 0.15 rel + 0.10 lead
        top_obj = top_sentences_objs[0]
        lead_b = 0.10 if any(best_excerpt == s for s in sentences[:3]) else 0.05
        evidence_strength = float(np_clip_val(0.45 * top_obj.sim_score + 0.30 * top_obj.ent_score + 0.15 * top_obj.score + lead_b, 0.45, 0.95))

        exec_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "supporting_sentences": top_sentences,
            "best_excerpt": best_excerpt,
            "evidence_strength": round(evidence_strength, 4),
            "sentence_ranking": [obj.to_dict() for obj in ranked_scores],
            "execution_ms": round(exec_ms, 2),
        }
