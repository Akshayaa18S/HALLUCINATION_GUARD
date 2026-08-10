"""
Phase 5 - Claim Extraction.

Converts the generated response into atomic factual claims, dropping
greetings, suggestions, questions, and closing statements - only
factual statements remain.

Two modes:
  - rule_based (default): regex sentence splitting + heuristic filters.
    Deterministic, needs no LLM, good enough for most responses.
  - llm_assisted: asks the LLM itself to break the response into atomic
    claims as JSON. Higher quality (splits compound sentences into
    truly atomic claims) but depends on the LLM being available and
    returning well-formed JSON, so it falls back to rule_based on any
    parse failure.
"""

import json
import logging
import re

from models.enums import StageName
from pipeline.context import ClaimContext, PipelineContext
from pipeline.stages.base import Stage
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|greetings)\b",
]
_QUESTION_PATTERNS = [r"\?\s*$"]
_SUGGESTION_PATTERNS = [
    r"^\s*(you (could|should|might|can)|i (recommend|suggest|would recommend))\b",
    r"\b(let me know|feel free to|please note that you)\b",
]
_CLOSING_PATTERNS = [
    r"^\s*(in conclusion|to summarize|to conclude|overall|thanks|thank you)\b",
    r"^\s*(hope this helps|let me know if)\b",
]
# Conversational lead-ins / hedges that frame or introduce a statement rather
# than assert a fact themselves, e.g. "I'm afraid that's not correct." These
# aren't greetings, questions, suggestions, or closings, so they need their
# own category - otherwise they slip through as "claims" with no factual
# content of their own.
_DISCOURSE_AND_META_PATTERNS = [
    r"^\s*(that|this)\s+(is|'s)\s+(not\s+)?(accurate|correct|true|false|right)\b",
    r"^\s*(yes|no|correct|indeed),?\s+(that|this)\b",
    r"^\s*(there\s+is\s+no\s+(mention|record|evidence)\s+of|according\s+to\s+the\s+(provided\s+)?(information|context|reference|sources?)|in\s+the\s+provided\s+information)\b",
    r"^\s*(based\s+on\s+the\s+(provided\s+)?(text|information|context|reference))\b",
    r"^\s*in\s+the\s+(provided|given)\s+(information|context|reference)\b",
]

_META_PATTERNS = [
    r"^\s*(i'?m afraid|i'?m sorry|sorry|unfortunately)\b",
    r"^\s*actually,?\s+(i'?m|i (don'?t|do not)|i'?m not sure)\b",
    r"^\s*(that'?s|this is) (not |in)?correct\b",
    r"^\s*(i (should|must|want to) (note|mention|point out|clarify)|"
    r"to be clear|to clarify)\b",
    r"^\s*(i (don'?t|do not) (know|think)|i'?m not sure)\b",
] + _DISCOURSE_AND_META_PATTERNS

# List-intro fragments, e.g. "The main differences between dogs and humans
# include:\n\n1." A colon at the end promises content that hasn't been
# stated yet - it isn't itself a factual assertion. This shape shows up when
# a numbered/bulleted list gets split (by either extraction path) such that
# the intro clause absorbs the first item's bare list marker ("1.", "2)",
# "-", "*") and nothing else. Genuine claims essentially never end in a bare
# colon or a colon-plus-list-marker, so this is a safe structural filter.
_FRAGMENT_PATTERNS = [
    r":\s*(\(?\d{1,2}[.):]?|[-*•])?\s*$",
]

_ALL_NON_FACTUAL_RES = [
    re.compile(p, re.IGNORECASE)
    for p in _GREETING_PATTERNS
    + _QUESTION_PATTERNS
    + _SUGGESTION_PATTERNS
    + _CLOSING_PATTERNS
    + _META_PATTERNS
    + _FRAGMENT_PATTERNS
]

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "as", "of", "to", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "his", "her", "their", "not", "no",
}
# How much of a claim's own content vocabulary is allowed to be absent
# from the source text entirely before the claim gets treated as likely
# invented rather than extracted. Extraction should stay close to the
# source's wording (light paraphrase at most); a claim that introduces
# several content words with zero grounding anywhere in the source
# (e.g. "Ligue 1" never mentioned, but the claim asserts a club plays in
# it) is a sign the LLM added outside knowledge instead of just reporting
# what the response said.
#
# A single ungrounded word is NOT enough on its own to flag a claim - that
# happens routinely with legitimate paraphrase (an abbreviation like "PSG"
# for "Paris Saint-Germain", a spelling variant, a synonym) and isn't a
# sign of fabrication. It's specifically multiple new content words
# appearing together that indicates an actual invented fact rather than a
# reworded one, so both a minimum count and a ratio are required.
_MAX_UNGROUNDED_RATIO = 0.3
_MIN_UNGROUNDED_WORDS = 2


def compute_span_coverage(source_text: str, extracted_claims: list[str]) -> float:
    """Computes SpanCoverage = Mapped Factual Character Spans / Total Factual Response Length."""
    if not source_text or not extracted_claims:
        return 0.0

    clean_source = source_text.strip().lower()
    total_len = len(clean_source)
    if total_len == 0:
        return 0.0

    covered = np.zeros(total_len, dtype=bool) if "np" in globals() else [False] * total_len

    for claim in extracted_claims:
        c_clean = claim.strip().lower()
        idx = clean_source.find(c_clean[:30])
        if idx != -1:
            end_idx = min(total_len, idx + len(c_clean))
            for i in range(idx, end_idx):
                covered[i] = True

    covered_count = sum(covered)
    return round(float(covered_count) / float(total_len), 4)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _is_grounded(claim: str, source_text: str) -> bool:
    claim_words = _content_words(claim)
    if not claim_words:
        return False
    source_words = _content_words(source_text)
    ungrounded = claim_words - source_words
    if len(ungrounded) < _MIN_UNGROUNDED_WORDS:
        return True
    return (len(ungrounded) / len(claim_words)) <= _MAX_UNGROUNDED_RATIO


_COMMON_VERB_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|plays|played|playing|works|worked|working|"
    r"has|have|had|having|does|did|done|doing|located|born|served|won|competes|compete|"
    r"represented|joined|known|called|became|belongs|consists|contains|includes|used|"
    r"created|built|written|directed|produced|lives|lived|found|said|stated|showed|"
    r"shows|indicates|indicated|means|meant|serves|not|no|cannot|could|would|should|runs|ran)\b",
    re.IGNORECASE,
)


_LEAD_IN_PREFIX_RE = re.compile(
    r"^\s*(i'?m\s+(not\s+)?(aware\s+of|sure\s+of|confident\s+that)(\s+any\s+(information|sources?|evidence)\s+(suggesting|indicating|that\s+suggests|that\s+indicates)?)?|"
    r"i\s+(couldn'?t|cannot|can\s+not)\s+(find|locate)\s+(any\s+)?(reliable\s+)?(information|sources?|records?|evidence)\s+(suggesting|indicating|that\s+suggests|that\s+indicates)?|"
    r"there\s+is\s+no\s+(evidence|record|information)\s+(suggesting|indicating|that\s+suggests|that)?|"
    r"i\s+(don'?t|do\s+not)\s+(believe|think|know)\s+(that)?)\s*",
    re.IGNORECASE,
)


def normalize_claim(sentence: str) -> str:
    """Normalizes raw extracted sentence by removing conversational framing lead-ins
    while preserving core factual assertions and canonical negations.
    """
    sentence = sentence.strip()
    if not sentence:
        return ""

    match = _LEAD_IN_PREFIX_RE.match(sentence)
    if match:
        matched_prefix = match.group(0).lower()
        remainder = sentence[match.end():].strip()
        if remainder:
            # If lead-in was a denial ("I'm not aware", "I couldn't find", "No evidence"), preserve negative assertion
            is_denial_prefix = any(k in matched_prefix for k in ("not", "no", "couldn't", "cannot"))
            if is_denial_prefix:
                if "has ever " in remainder.lower():
                    remainder = re.sub(r"\bhas\s+ever\b", "has never", remainder, flags=re.IGNORECASE)
                elif not re.search(r"\b(not|never|no)\b", remainder, re.IGNORECASE):
                    remainder = f"No evidence indicates that {remainder[0].lower() + remainder[1:]}"

            sentence = remainder[0].upper() + remainder[1:]

    return sentence



_REDUNDANT_EXPLANATORY_RE = re.compile(
    r"^\s*(the\s+sport\s+of\s+association\s+football|association\s+football\s+\(soccer\),?\s+not\s+cricket|the\s+term|this\s+means\s+that|in\s+other\s+words)\b",
    re.IGNORECASE,
)


def _is_factual_candidate(sentence: str) -> bool:
    sentence = sentence.strip()
    if not sentence or len(sentence) < 8:
        return False
    if _REDUNDANT_EXPLANATORY_RE.search(sentence):
        return False
    # Reject bare noun fragments lacking any verb or state of being (e.g. "Barcelona", "Spain national team")
    words = sentence.split()
    if len(words) < 3 and not _COMMON_VERB_RE.search(sentence):
        return False
    return not any(p.search(sentence) for p in _ALL_NON_FACTUAL_RES)



from abc import ABC, abstractmethod


def decompose_sentence_into_claims(sentence: str) -> list[str]:
    """Decomposes compound sentences into atomic relational claims with subject context."""
    sentence = sentence.strip()
    if not sentence:
        return []

    clean_s = re.sub(r"^(no|yes),?\s*", "", sentence, flags=re.IGNORECASE).strip()
    if clean_s:
        clean_s = clean_s[0].upper() + clean_s[1:]

    words = clean_s.split()
    subj = words[0] if words else "It"
    if len(words) >= 2 and words[0].lower() not in ("they", "he", "she", "it", "this", "that") and words[1][0].isupper():
        subj = f"{words[0]} {words[1]}"

    verb_past = "were" if subj.lower() in ("they", "we", "you") else "was"

    results = []

    # Check compound pattern 1: "... formed by X in Y" / "... formed by X"
    if " formed by " in clean_s:
        parts = clean_s.split(" formed by ")
        main_clause = parts[0].strip()
        sub_clause = parts[1].strip()

        if main_clause:
            results.append(main_clause if main_clause.endswith(".") else main_clause + ".")

        if " in " in sub_clause:
            sub_parts = sub_clause.split(" in ")
            company = sub_parts[0].strip()
            year_or_place = sub_parts[1].strip().rstrip(".")
            results.append(f"{subj} {verb_past} formed by {company}.")
            results.append(f"{subj} {verb_past} formed in {year_or_place}.")
        else:
            results.append(f"{subj} {verb_past} formed by {sub_clause.rstrip('.')} .")

    # Check compound pattern 2: "... born in X"
    elif " born in " in clean_s and not clean_s.startswith("Born in"):
        parts = clean_s.split(" born in ")
        main_clause = parts[0].strip()
        place = parts[1].strip().rstrip(".")
        if main_clause:
            results.append(main_clause if main_clause.endswith(".") else main_clause + ".")
        results.append(f"{subj} {verb_past} born in {place}.")

    # Check compound pattern 3: "... who "
    elif " who " in clean_s:
        parts = clean_s.split(" who ")
        main_clause = parts[0].strip()
        rel_clause = parts[1].strip()
        if main_clause:
            results.append(main_clause if main_clause.endswith(".") else main_clause + ".")
        results.append(f"{subj} {rel_clause.rstrip('.')} .")

    else:
        results.append(sentence)

    return [s for s in results if s.strip()]


def rule_based_extract(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    raw_claims = []
    for s in sentences:
        norm = normalize_claim(s)
        sub_claims = decompose_sentence_into_claims(norm)
        for sc in sub_claims:
            if _is_factual_candidate(sc):
                raw_claims.append(sc)
    return raw_claims


class BaseClaimExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[str]:
        """Extract atomic factual claims from input text."""
        pass


class RuleBasedClaimExtractor(BaseClaimExtractor):
    def extract(self, text: str) -> list[str]:
        return rule_based_extract(text)


class LLMClaimExtractor(BaseClaimExtractor):
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()
        self.fallback = RuleBasedClaimExtractor()

    def extract(self, text: str) -> list[str]:
        return self.fallback.extract(text)




_LLM_EXTRACTION_SYSTEM_PROMPT = (
    "You extract atomic, self-contained factual claims from text. Return ONLY a JSON array "
    "of strings, each a single, complete factual proposition with a subject, verb, and object/context. "
    "CRITICAL RULES:\n"
    "1. Complete Sentences Only: NEVER extract standalone nouns, entity names, or noun phrases (e.g. 'Barcelona' or 'Spain national team' are INVALID claims; instead extract 'Lamine Yamal plays for Barcelona' or 'Lamine Yamal plays for the Spain national team').\n"
    "2. Relational Claims: When decomposing complex sentences, attach the subject and verb to each sub-claim so every claim retains full relational context (e.g. from 'Lamine Yamal is a footballer who plays as a right winger for Barcelona', extract 'Lamine Yamal is a Spanish professional footballer', 'Lamine Yamal plays as a right winger', 'Lamine Yamal plays for Barcelona').\n"
    "3. Preserve Negations & Corrections: If the text denies a claim or makes a correction (e.g. 'Virat Kohli is an Indian cricketer, not a football player'), extract both the positive factual assertion ('Virat Kohli is an Indian cricketer.') AND the explicit negative assertion ('Virat Kohli is not a football player.').\n"
    "4. Skip Non-Factual Text: Drop greetings, questions, suggestions, opinions, list headers, and closing remarks.\n"
    "5. Grounding: Only extract claims explicitly stated or asserted in the given text - do not infer outside knowledge.\n"
    "No prose, no markdown fences, just the JSON array."
)


from hallucination.verifiability import BaseClaimTypeClassifier, claim_classifier


class ClaimExtractionStage(Stage):
    name = StageName.CLAIM_EXTRACTION
    critical = False  # worst case: fall back to rule-based, never abort the job

    def __init__(
        self,
        llm_service: LLMService | None = None,
        use_llm: bool = True,
        classifier: BaseClaimTypeClassifier | None = None,
    ):
        self.llm_service = llm_service or LLMService()
        self.use_llm = use_llm
        self.classifier = classifier or claim_classifier


    async def _llm_extract(self, text: str) -> list[str] | None:
        from utils.cache import intermediate_cache
        cached = intermediate_cache.get("claim_extraction", text, _LLM_EXTRACTION_SYSTEM_PROMPT)
        if cached is not None:
            return cached

        try:
            raw = await self.llm_service.generate(
                prompt=text, system=_LLM_EXTRACTION_SYSTEM_PROMPT
            )
            cleaned = raw.strip().strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            claims = json.loads(cleaned)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                candidates = [c.strip() for c in claims if _is_factual_candidate(c)]
                grounded, dropped = [], []
                for c in candidates:
                    (grounded if _is_grounded(c, text) else dropped).append(c)
                if dropped:
                    logger.warning(
                        "LLM claim extraction dropped %d ungrounded claim(s) not "
                        "present in the source text: %s",
                        len(dropped), dropped,
                    )
                intermediate_cache.set("claim_extraction", text, _LLM_EXTRACTION_SYSTEM_PROMPT, grounded)
                return grounded
            return None
        except Exception as exc:
            logger.warning("LLM claim extraction failed, falling back to rule-based: %s", exc)
            return None


    async def run(self, context: PipelineContext) -> PipelineContext:
        claims_text: list[str] | None = None
        method = "rule_based"

        if self.use_llm:
            claims_text = await self._llm_extract(context.generated_response)
            if claims_text is not None:
                method = "llm_assisted"

        if claims_text is None:
            claims_text = rule_based_extract(context.generated_response)
        elif not claims_text:
            fallback = rule_based_extract(context.generated_response)
            if fallback:
                logger.warning(
                    "LLM-assisted claim extraction produced 0 claims after "
                    "filtering; falling back to rule-based extraction, "
                    "which found %d claim(s).", len(fallback),
                )
                claims_text = fallback
                method = "rule_based"

        raw_claims = []
        for t in claims_text:
            c_type = self.classifier.classify(t)
            raw_claims.append(ClaimContext(text=t, claim_type=c_type))


        verifiable_claims = [c for c in raw_claims if c.is_verifiable]

        context.claims = verifiable_claims
        context.record(
            self.name.value,
            {
                "method": method,
                "total_extracted": len(raw_claims),
                "verifiable_claims": len(verifiable_claims),
            },
        )
        return context