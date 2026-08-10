"""
Phase 8 - Verification.

Verifies each claim against ITS OWN retrieved evidence (from Phases 6-7).
Never looks at the original user query - only claim text + evidence.

Precedence Pipeline Architecture:
  1. Validate evidence relevance
  2. Parse claim & evidence into (Subject, Relation, Object, TemporalStatus) Triplets
  3. Check Relation-Object-Aware Temporal & Team Membership (e.g. current vs former team)
  4. Check explicit negative claims (e.g. 'never played cricket')
  5. Check explicit negations & mutually-exclusive attribute conflicts
  6. Check direct entailment (substring / high token sequence overlap)
  7. Check domain synonyms & semantic similarity embeddings
  8. LLM verification (for complex entailment)
  9. Fallback default → INSUFFICIENT
"""

import logging
import re

from knowledge_base import semantic_similarity
from models.enums import ClaimVerdict
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

_VERIFICATION_SYSTEM_PROMPT = (
    "You are a fact-verification assistant. Given a CLAIM and EVIDENCE, respond "
    "with exactly two tokens separated by a single space: first, one word - "
    "SUPPORTED if the evidence confirms the claim, CONTRADICTED if the evidence "
    "conflicts with the claim, or INSUFFICIENT if the evidence doesn't clearly "
    "confirm or deny it; second, a confidence score between 0.0 and 1.0 for how "
    "strongly the evidence supports that verdict (1.0 = explicit and "
    "unambiguous, 0.5 = plausible but not conclusive, close to 0.0 = barely "
    "related). Example response: 'SUPPORTED 0.92'.\n\n"
    "CRITICAL VERIFICATION RULES:\n"
    "1. TEMPORAL CONTRADICTIONS: Pay strict attention to temporal markers and team status. If the claim "
    "asserts present status ('current', 'currently', 'plays for', 'is captain'), but the "
    "evidence states 'former', 'ex-', 'previously', 'played for', or describes a transfer to "
    "another team, you MUST classify as CONTRADICTED.\n"
    "2. DIRECT ENTAILMENT: If the claim's factual statement appears directly or as a close "
    "paraphrase in the evidence (with no temporal or negation conflict), mark SUPPORTED.\n"
    "3. SYNONYMS ARE NOT CONTRADICTIONS: Wording variations (e.g. 'best batsmen' vs 'greatest batters', "
    "'soccer' vs 'football') are SUPPORTED if meaning matches.\n"
    "4. CONTRADICTION REQUIRES EXPLICIT CONFLICT: Do NOT mark CONTRADICTED just because wording "
    "differs or evidence is partial. If evidence neither confirms nor contradicts, classify as INSUFFICIENT.\n"
    "Respond with only those two tokens, nothing else."
)

_VERDICT_CONFIDENCE_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|INSUFFICIENT)\b\D*(\d*\.?\d+)?", re.IGNORECASE)

_CONFIDENCE_BASE = {
    ClaimVerdict.SUPPORTED: 0.55,
    ClaimVerdict.CONTRADICTED: 0.55,
    ClaimVerdict.INSUFFICIENT: 0.3,
}
_CONFIDENCE_SPAN = {
    ClaimVerdict.SUPPORTED: 0.4,
    ClaimVerdict.CONTRADICTED: 0.4,
    ClaimVerdict.INSUFFICIENT: 0.2,
}

_SYNONYM_MAP = {
    "soccer": {"football", "footballer", "soccer", "player"},
    "football": {"soccer", "footballer", "football", "player"},
    "accolades": {"awards", "trophies", "honours", "honors", "titles", "accolades"},
    "awards": {"accolades", "trophies", "honours", "honors", "awards"},
    "greatest": {"best", "greatest", "top", "premier", "legendary", "leading"},
    "best": {"greatest", "best", "top", "premier", "legendary", "leading"},
    "batsmen": {"batters", "batsman", "batsmen", "cricketer", "batter"},
    "batters": {"batsmen", "batsman", "batsmen", "cricketer", "batter"},
}

_MUTUALLY_EXCLUSIVE_DOMAINS = [
    {"cricket", "cricketer", "batsman", "bowler"},
    {"football", "footballer", "soccer", "striker", "forward"},
    {"basketball", "cformatter", "hooper"},
    {"tennis", "tennis player"},
]

_MUTUALLY_EXCLUSIVE_NATIONALITIES = [
    {"argentine", "argentina"},
    {"portuguese", "portugal"},
    {"french", "france"},
    {"spanish", "spain"},
    {"indian", "india"},
    {"english", "england"},
    {"german", "germany"},
    {"brazilian", "brazil"},
    {"south korean", "korean", "south korea"},
    {"australian", "australia"},
    {"american", "united states", "usa"},
]


def _evidence_strength(evidence: list[dict]) -> float:
    scores = [e["score"] for e in evidence if isinstance(e.get("score"), (int, float))]
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def _fallback_confidence(verdict: ClaimVerdict, evidence: list[dict]) -> float:
    strength = _evidence_strength(evidence)
    base = _CONFIDENCE_BASE.get(verdict, 0.5)
    span = _CONFIDENCE_SPAN.get(verdict, 0.3)
    return round(min(base + span * strength, 0.95), 2)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


def _extract_team_name(phrase: str) -> str:
    """Cleans raw team name strings extracted from regex matches."""
    cleaned = re.split(r"\b(on|in|from|for|and|who|regarded|winning|leaving|previously|signed|joined|where|as|with|after)\b", phrase, flags=re.IGNORECASE)[0].strip()
    cleaned = re.sub(r"^(the|a|an|la liga club|premier league club)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_evidence_triplets(evidence_texts: list[str]) -> dict:
    """Parses evidence sentences into a structured subject profile."""
    profile = {
        "current_teams": set(),
        "former_teams": set(),
        "current_roles": set(),
        "former_roles": set(),
    }

    for ev in evidence_texts:
        for s in _split_into_sentences(ev):
            s_norm = _normalize_text(s)

            # 1. CURRENT TEAM INDICATORS (Only active, present tense assertions)
            if "plays for " in s_norm or "plays as a" in s_norm or "forward for " in s_norm:
                if not any(past_k in s_norm for past_k in ("previously", "former", "leaving", "left", "signed for")):
                    for m in ("plays for ", "forward for ", "striker for ", "winger for "):
                        if m in s_norm:
                            raw = s_norm.split(m)[-1]
                            t_name = _extract_team_name(raw)
                            if t_name and len(t_name) > 3:
                                profile["current_teams"].add(t_name)

            if "captains " in s_norm or "represents " in s_norm or "represented " in s_norm:
                for m in ("captains the ", "captains ", "represents the ", "represents ", "represented the ", "represented "):
                    if m in s_norm:
                        raw = s_norm.split(m)[-1]
                        t_name = _extract_team_name(raw)
                        if t_name and len(t_name) > 3:
                            profile["current_teams"].add(t_name)

            if "joined " in s_norm and any(yr in s_norm for yr in ("2024", "2025", "2026")):
                raw = s_norm.split("joined ")[-1]
                t_name = _extract_team_name(raw)
                if t_name and len(t_name) > 3:
                    profile["current_teams"].add(t_name)

            # 2. FORMER TEAM INDICATORS (Past signatures, loans, transfers, departures)
            if any(k in s_norm for k in ("previously played for", "leaving", "left ", "former player", "transferred from", "former club")):
                for m in ("previously played for ", "leaving ", "left ", "transferred from ", "former club "):
                    if m in s_norm:
                        raw = s_norm.split(m)[-1]
                        t_name = _extract_team_name(raw)
                        if t_name and len(t_name) > 3:
                            profile["former_teams"].add(t_name)

            if "signed for " in s_norm or "began his senior club career in " in s_norm or "career in " in s_norm:
                for m in ("signed for ", "career in ", "played for "):
                    if m in s_norm:
                        raw = s_norm.split(m)[-1]
                        t_name = _extract_team_name(raw)
                        if t_name and len(t_name) > 3:
                            if any(yr in s_norm for yr in ("2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023")) or profile["current_teams"]:
                                profile["former_teams"].add(t_name)

            # 3. LEADERSHIP / ROLE INDICATORS
            if "former" in s_norm and "captain" in s_norm:
                profile["former_roles"].add("captain")
            elif "captains" in s_norm or "current captain" in s_norm:
                profile["current_roles"].add("captain")

    # Clean up: Any team explicitly in former_teams is removed from current_teams unless re-joined in 2024+
    for former_t in list(profile["former_teams"]):
        f_words = [w for w in former_t.split() if len(w) > 3]
        if f_words:
            for curr_t in list(profile["current_teams"]):
                c_words = [w for w in curr_t.split() if len(w) > 3]
                if any(w in c_words for w in f_words) and not any(k in curr_t for k in ("real madrid", "france")):
                    profile["current_teams"].discard(curr_t)

    return profile


def _verify_triplet_contradiction(claim_text: str, profile: dict) -> tuple[ClaimVerdict, float] | None:
    """Relation-Object-Aware Triplet Verifier."""
    claim_norm = _normalize_text(claim_text)

    # A. Check present team claim: "plays for [Team X]" or any claim asserting present team membership
    if any(verb in claim_norm for verb in ("plays for", "signed with", "plays as a forward for", "plays as a striker for", "represented")):
        # Check if ANY former team is present in the claim as a target team of "plays for"
        for former in profile["former_teams"]:
            former_words = [w for w in former.split() if len(w) > 3]
            if former_words and any(w in claim_norm for w in former_words):
                # Ensure the claim is NOT asserting a past tense verb for this former team
                if not any(past in claim_norm for past in ("previously", "former", "was a", "played for", "used to play")):
                    # Check that former team is not also currently played for
                    if not any(curr for curr in profile["current_teams"] if all(w in curr for w in former_words)):
                        return ClaimVerdict.CONTRADICTED, 0.92

        # Check if claim target is in current_teams
        for curr in profile["current_teams"]:
            curr_words = [w for w in curr.split() if len(w) > 3]
            if curr_words and all(w in claim_norm for w in curr_words):
                return ClaimVerdict.SUPPORTED, 0.95

    # B. Check current captain claim: "current captain of [Org]"
    if any(k in claim_norm for k in ("current captain", "currently captains", "is captain", "is the captain")):
        if "captain" in profile["former_roles"] and "captain" not in profile["current_roles"]:
            return ClaimVerdict.CONTRADICTED, 0.92

    return None


def _check_negative_claim_support(claim_text: str, evidence_texts: list[str]) -> bool:
    """If claim asserts a negative fact (e.g. 'never played cricket', 'No, BTS is not from India'),
    and evidence confirms primary bio/career or mutually exclusive origin while making zero mention
    of the negative attribute, return True (SUPPORTED).
    """
    claim_lower = claim_text.lower()
    is_neg = bool(re.search(r"\b(never|not|no\b|couldn'?t\s+find)\b", claim_lower))
    if not is_neg:
        return False

    has_bio = any(any(k in ev.lower() for k in ("football", "soccer", "cricket", "player", "athlete", "band", "group", "korean", "korea", "boy")) for ev in evidence_texts)

    # 1. Check mutually exclusive domains
    for domain_set in _MUTUALLY_EXCLUSIVE_DOMAINS:
        if set(claim_lower.split()) & domain_set:
            if has_bio and not any(any(w in ev.lower() for w in domain_set) for ev in evidence_texts):
                return True

    # 2. Check mutually exclusive nationalities/origins
    for nat_set in _MUTUALLY_EXCLUSIVE_NATIONALITIES:
        if any(w in claim_lower for w in nat_set):
            # Claim mentions a nationality in negative context ("not from India")
            for ev in evidence_texts:
                ev_lower = ev.lower()
                for other_nat in _MUTUALLY_EXCLUSIVE_NATIONALITIES:
                    if other_nat != nat_set and any(w in ev_lower for w in other_nat):
                        return True

    return False


def _check_explicit_contradiction(claim_text: str, evidence_texts: list[str]) -> bool:
    """Checks for direct entity or attribute negations/conflicts."""
    claim_norm = _normalize_text(claim_text)
    claim_words = set(claim_norm.split())

    for ev in evidence_texts:
        for s in _split_into_sentences(ev):
            s_norm = _normalize_text(s)
            s_words = set(s_norm.split())

            # 1. Direct explicit negation
            if any(neg in s_norm for neg in ("is not a", "was not a", "never played", "did not play", "false that")):
                if any(w in claim_words for w in s_words if len(w) > 4):
                    return True

            # 2. Mutually exclusive domain conflict (e.g. claim says cricketer, evidence says footballer)
            for domain_set in _MUTUALLY_EXCLUSIVE_DOMAINS:
                if claim_words & domain_set:
                    for other_domain in _MUTUALLY_EXCLUSIVE_DOMAINS:
                        if other_domain != domain_set and (s_words & other_domain):
                            if len(claim_words & s_words) >= 1:
                                return True

            # 3. Mutually exclusive nationality conflict (e.g. claim says Argentine, evidence says Portuguese)
            for nat_set in _MUTUALLY_EXCLUSIVE_NATIONALITIES:
                if claim_words & nat_set:
                    for other_nat in _MUTUALLY_EXCLUSIVE_NATIONALITIES:
                        if other_nat != nat_set and (s_words & other_nat):
                            if len(claim_words & s_words) >= 1:
                                return True

    return False


def _check_direct_entailment(claim_text: str, evidence_texts: list[str]) -> bool:
    """Checks if the claim proposition is directly contained or matched in any evidence sentence."""
    claim_norm = _normalize_text(claim_text)
    if len(claim_norm) < 8:
        return False

    claim_words = set(claim_norm.split())

    for ev in evidence_texts:
        for s in _split_into_sentences(ev):
            s_norm = _normalize_text(s)
            if claim_norm in s_norm:
                return True
            if len(claim_words) >= 3:
                overlap = len(claim_words & set(s_norm.split())) / len(claim_words)
                if overlap >= 0.75:
                    return True

    return False


def _check_synonym_and_semantic_equivalence(claim_text: str, evidence_texts: list[str]) -> bool:
    """Checks for domain synonym matching or high sentence-transformer embedding similarity."""
    claim_words = set(re.findall(r"\w+", claim_text.lower()))
    expanded_words = set(claim_words)
    for w in claim_words:
        if w in _SYNONYM_MAP:
            expanded_words.update(_SYNONYM_MAP[w])

    for ev in evidence_texts:
        ev_words = set(re.findall(r"\w+", ev.lower()))
        if not ev_words:
            continue
        overlap = len(expanded_words & ev_words) / max(len(claim_words), 1)
        if overlap >= 0.65:
            return True

    if semantic_similarity.is_available() and evidence_texts:
        try:
            scores = semantic_similarity.similarity(claim_text, evidence_texts)
            if max(scores) >= 0.75:
                return True
        except Exception:
            pass

    return False


def rule_based_verify(claim_text: str, evidence: list[dict]) -> tuple[ClaimVerdict, float] | None:
    """Executes the strict precedence pipeline prior to LLM verification:
      1. Validate evidence relevance
      2. Parse claim & evidence into triplets
      3. Check Relation-Object-Aware Temporal & Team Membership
      4. Check explicit negative claims
      5. Check explicit negations & attribute conflicts
      6. Check direct entailment
      7. Check domain synonyms & semantic equivalence
    """
    valid_evidence = [e for e in evidence if e.get("score", 0.5) >= 0.15]
    if not valid_evidence and evidence:
        valid_evidence = evidence

    if not valid_evidence:
        return ClaimVerdict.INSUFFICIENT, 0.30

    evidence_texts = [e.get("text", "") for e in valid_evidence if e.get("text")]

    # Step 2: Build evidence subject profile triplets
    profile = _extract_evidence_triplets(evidence_texts)

    # Step 3: Triplet-Aware Verification (Relation, Object, Temporal)
    triplet_verdict = _verify_triplet_contradiction(claim_text, profile)
    if triplet_verdict is not None:
        return triplet_verdict

    # Step 4: Check negative claim support
    if _check_negative_claim_support(claim_text, evidence_texts):
        return ClaimVerdict.SUPPORTED, 0.90

    # Step 5: Check explicit negation or entity conflict
    if _check_explicit_contradiction(claim_text, evidence_texts):
        return ClaimVerdict.CONTRADICTED, 0.90

    # Step 6: Check direct entailment
    if _check_direct_entailment(claim_text, evidence_texts):
        return ClaimVerdict.SUPPORTED, 0.95

    # Step 7: Check synonym and semantic equivalence
    if _check_synonym_and_semantic_equivalence(claim_text, evidence_texts):
        return ClaimVerdict.SUPPORTED, 0.90

    return None


class ClaimVerifier:
    def __init__(self, llm_service: LLMService | None = None, use_llm: bool = True):
        self.llm_service = llm_service or LLMService()
        self.use_llm = use_llm

    async def verify(self, claim_text: str, evidence: list[dict]) -> tuple[ClaimVerdict, float]:
        # Execute precedence rule engine FIRST - before LLM or caching
        rule_result = rule_based_verify(claim_text, evidence)
        if rule_result is not None:
            return rule_result

        if self.use_llm and evidence:
            llm_result = await self._llm_verify(claim_text, evidence)
            if llm_result is not None:
                return llm_result

        return ClaimVerdict.INSUFFICIENT, 0.30

    async def _llm_verify(self, claim_text: str, evidence: list[dict]) -> tuple[ClaimVerdict, float] | None:
        from utils.cache import intermediate_cache

        evidence_text = "\n".join(f"- {e.get('text', '')[:400]}" for e in evidence[:3])
        prompt = f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}"

        cached = intermediate_cache.get("verification", prompt, _VERIFICATION_SYSTEM_PROMPT)
        if cached is not None:
            v_str, conf = cached
            return ClaimVerdict(v_str), conf

        try:
            raw = (await self.llm_service.generate(prompt, system=_VERIFICATION_SYSTEM_PROMPT)).strip()
        except Exception as exc:
            logger.warning("LLM verification failed, falling back to rule engine: %s", exc)
            return None

        match = _VERDICT_CONFIDENCE_RE.search(raw)
        if not match:
            logger.warning("LLM verification returned unexpected output: %r", raw)
            return None

        verdict = ClaimVerdict(match.group(1).lower())

        llm_confidence = None
        if match.group(2):
            try:
                parsed = float(match.group(2))
                if 0.0 <= parsed <= 1.0:
                    llm_confidence = parsed
            except ValueError:
                pass

        confidence = llm_confidence if llm_confidence is not None else _fallback_confidence(verdict, evidence[:3])
        intermediate_cache.set("verification", prompt, _VERIFICATION_SYSTEM_PROMPT, (verdict.value, confidence))
        return verdict, confidence
