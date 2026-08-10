"""
Phase 7 - retrieval priority + evidence ranking.

Priority order for which entity gets looked up / ranked first:
    PERSON -> ORGANIZATION -> LOCATION/COUNTRY/CITY -> generic context

This is the fix for the "never search Africa before Lamine Yamal" bug
described in the spec: continents/regions are LOCATION, which ranks
below PERSON and ORGANIZATION, so a claim mentioning both a person and
a location always resolves the person's entity first.
"""

import logging
import re

from knowledge_base import semantic_similarity
from pipeline.context import Entity

logger = logging.getLogger(__name__)

_PRIORITY = {
    "PERSON": 0,
    "ORGANIZATION": 1,
    "SPORTS_TEAM": 1,
    "LOCATION": 2,
    "COUNTRY": 2,
    "CITY": 2,
    "EVENT": 3,
    "PRODUCT": 3,
    "DATE": 4,
    "NUMBER": 5,
}


def priority_order(entities: list[Entity]) -> list[Entity]:
    """Entities sorted by retrieval priority, PERSON first."""
    return sorted(entities, key=lambda e: _PRIORITY.get(e.label, 99))


def normalize_evidence_text(text: str) -> str:
    """Normalizes evidence snippet for deduplication."""
    if not text:
        return ""
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+([.!?,;])", r"\1", text)
    text = re.sub(r"([.!?,;])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text.lower()).strip()



from config.settings import settings


def _is_disambiguation_page(title: str, text: str) -> bool:
    """Detects disambiguation pages that do not contain core factual evidence."""
    norm = ((title or "") + " " + (text or "")).lower()
    return "disambiguation" in norm or "may refer to:" in norm or "may refer to " in norm


_DERIVATIVE_TERMS = {"film", "movie", "season", "discography", "album", "list of", "tribute"}
_TANGENTIAL_PATTERNS = re.compile(
    r"(rivalry|[\u2013\-]ronaldo|[\u2013\-]messi|\(\d{4}\s+film\)|national.+team\s+(record|cap|stat))",
    re.IGNORECASE,
)



def _canonical_score_adjustment(claim_text: str, title: str, base_score: float) -> float:
    """Boosts canonical main entity biography pages and penalizes derivative pages
    (films, lists, seasons) unless the claim explicitly asks about them.
    """
    if not title:
        return base_score

    title_lower = title.strip().lower()
    claim_lower = claim_text.lower()

    # Extract proper noun subjects from claim (e.g. "Lionel Messi" or "Lamine Yamal")
    subjects = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim_text)
    subj_lowers = [s.lower() for s in subjects]

    adjusted = base_score

    # Canonical title match boost (requires exact full subject match or multi-word entity match)
    if any(title_lower == s for s in subj_lowers) or (len(title_lower.split()) > 1 and title_lower in claim_lower):
        adjusted += 0.25


    # Derivative page penalty (e.g. "Messi (2014 film)", "2025-26 FC Barcelona season")
    for term in _DERIVATIVE_TERMS:
        if term in title_lower and term not in claim_lower:
            adjusted -= 0.20

    return max(0.0, min(1.0, adjusted))


def rank_evidence(claim_text: str, evidence: list[dict], top_k: int, min_score: float | None = None) -> list[dict]:
    """Rank evidence snippets by relevance to the claim text, deduplicate by
    canonical URL / title and normalized snippet text, filter out disambiguation
    pages, boost canonical primary entity pages, and filter weak matches below min_score.
    """
    if not evidence:
        return []

    threshold = min_score if min_score is not None else getattr(settings, "min_evidence_score", 0.10)

    # Filter out disambiguation pages upfront unless it's the only evidence
    filtered_evidence = [e for e in evidence if not _is_disambiguation_page(e.get("title", ""), e.get("text", ""))]
    if not filtered_evidence:
        filtered_evidence = evidence

    texts = [e.get("text", "") for e in filtered_evidence]

    if semantic_similarity.is_available():
        raw_scores = semantic_similarity.similarity(claim_text, texts)
    else:
        claim_words = set(claim_text.lower().split())
        raw_scores = []
        for t in texts:
            words = set(t.lower().split())
            union = claim_words | words
            raw_scores.append(len(claim_words & words) / len(union) if union else 0.0)

    by_key = {}
    for e, r_score in zip(filtered_evidence, raw_scores):
        score = _canonical_score_adjustment(claim_text, e.get("title", ""), float(r_score))
        if score < threshold:
            continue
        item = dict(e)
        item["score"] = round(score, 4)
        canon_url = (item.get("url") or item.get("title") or "").strip().lower()
        norm_txt = normalize_evidence_text(item.get("text", ""))[:300]
        key = (canon_url, norm_txt)
        if key not in by_key or item["score"] > by_key[key]["score"]:
            by_key[key] = item

    # Tangential page suppression: if a high-relevance canonical page exists, drop weak tangential pages
    subjects = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim_text)
    subj_lowers = [s.lower() for s in subjects]
    has_high_canonical = any(
        e.get("score", 0) >= 0.60 and any(s == (e.get("title") or "").lower() for s in subj_lowers)
        for e in by_key.values()
    )
    if has_high_canonical:
        by_key = {k: v for k, v in by_key.items() if v.get("score", 0) >= max(threshold, 0.20)}

    ranked = sorted(by_key.values(), key=lambda x: x["score"], reverse=True)

    # Entity-relevance filtering: when a canonical biography is the top result,
    # drop pages that are single-word name fragments, season articles, or records
    # pages — they don't directly support or refute claims about the entity.
    if ranked and ranked[0].get("score", 0) >= 0.80:
        top_title = (ranked[0].get("title") or "").lower()
        if any(s in top_title or top_title in s for s in subj_lowers):
            filtered = [ranked[0]]
            for e in ranked[1:]:
                title_low = (e.get("title") or "").lower()
                # Skip single-word first-name fragments (e.g. "Lamine")
                if len(title_low.split()) == 1 and any(title_low in s for s in subj_lowers):
                    continue
                # Skip season/records pages (e.g. "2023-24 FC Barcelona season")
                if re.search(r"\d{4}[–\-]\d{2,4}\b", title_low) or "records" in title_low:
                    continue
                # Skip rivalry, film, and generic national team stats pages
                if _TANGENTIAL_PATTERNS.search(title_low):
                    continue
                # Keep if high score, or title mentions the primary entity
                if e.get("score", 0) >= 0.50 or any(s in title_low for s in subj_lowers):
                    filtered.append(e)
            return filtered[:top_k]


    return ranked[:top_k]










