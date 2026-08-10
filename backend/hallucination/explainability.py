"""
Phase 10 - Explainability.

Turns the per-claim verdicts into the final human-readable output:
verified_answer, explanation, evidence list, and contradictions.
"""

from dataclasses import dataclass
import re

from models.enums import ClaimVerdict



@dataclass
class ClaimExplainInput:
    text: str
    verdict: ClaimVerdict
    confidence: float | None
    evidence: list[dict]
    # See pipeline.stages.query_consistency - True when this claim's entity
    # looks like a stand-in the model invented after failing to recall the
    # entity the query actually asked about.
    fabricated_alternative: bool = False


@dataclass
class ExplainabilityResult:
    verified_answer: str
    explanation: str
    contradictions: list[dict]


def _clean_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:]


def _extract_subject_prefix(sentence: str) -> str:
    words = sentence.split()
    if not words or words[0].lower() in ("the", "there", "according", "this", "that", "in", "based", "it"):
        return ""
    cap_words = []
    for w in words:
        clean_w = w.rstrip(".!?,")
        if clean_w in ("is", "was", "are", "plays", "has", "captains", "signed", "represented"):
            break
        if clean_w and (clean_w[0].isupper() or clean_w in ("de", "da", "von", "van")):
            cap_words.append(clean_w)
        else:
            break
    res = " ".join(cap_words)
    return res if len(res.split()) >= 1 and res.lower() not in ("the", "there", "this", "that") else ""



def _synthesize_supported_claims(sentences: list[str]) -> list[str]:
    if not sentences:
        return []

    clean_sentences = []

    for s in sentences:
        cs = _clean_sentence(s).rstrip(".!?")
        if not cs or any(m in cs.lower() for m in ("reference information", "provided information", "according to", "there is no mention", "that is not accurate", "the reference")):
            continue
        clean_sentences.append(cs)

    if not clean_sentences:
        return []

    subjs = [_extract_subject_prefix(s) for s in clean_sentences if _extract_subject_prefix(s)]
    canonical_subj = max(subjs, key=len) if subjs else ""

    is_clauses = []
    plays_as = []
    plays_for = []
    captains_for = []
    other_clauses = []

    for s in clean_sentences:
        rem = s[len(canonical_subj):].strip() if canonical_subj and s.startswith(canonical_subj) else s
        for sub in subjs:
            if s.startswith(sub):
                rem = s[len(sub):].strip()
                break

        rem_lower = rem.lower()
        if rem_lower.startswith("is ") or rem_lower.startswith("was ") or rem_lower.startswith("are "):
            is_clauses.append(rem)
        elif rem_lower.startswith("plays as "):
            plays_as.append(rem[len("plays as "):].strip())
        elif rem_lower.startswith("captains "):
            team_name = rem[len("captains "):].strip()
            if team_name.lower().startswith("both "):
                team_name = team_name[5:].strip()
            captains_for.append(team_name)
        elif rem_lower.startswith("plays for "):
            team_name = rem[len("plays for "):].strip()
            if team_name.lower() == "barcelona":
                team_name = "FC Barcelona"
            plays_for.append(team_name)
        elif rem_lower.startswith("has played for "):
            team_name = rem[len("has played for "):].strip()
            if team_name.lower() == "barcelona":
                team_name = "FC Barcelona"
            plays_for.append(team_name)
        elif rem_lower.startswith("plays "):
            plays_for.append(rem[len("plays "):].strip())
        else:
            if len(rem.split()) >= 2:
                other_clauses.append(rem)

    def _dedup(items):
        seen = set()
        res = []
        for x in items:
            if x.lower() not in seen:
                seen.add(x.lower())
                res.append(x)
        return res

    is_clauses = _dedup(is_clauses)
    plays_as = _dedup(plays_as)
    plays_for = _dedup(plays_for)
    captains_for = _dedup(captains_for)

    parts = []
    if is_clauses:
        join_is = " and ".join(is_clauses)
        parts.append(f"{canonical_subj} {join_is}" if canonical_subj else join_is)
    elif canonical_subj:
        parts.append(canonical_subj)

    rel_parts = []
    all_teams = _dedup(plays_for + captains_for)

    if plays_as and all_teams:
        if captains_for and plays_for:
            rel_parts.append(f"plays as {' and '.join(plays_as)} for and captains {' and '.join(all_teams)}")
        elif captains_for:
            rel_parts.append(f"plays as {' and '.join(plays_as)} and captains {' and '.join(all_teams)}")
        else:
            rel_parts.append(f"plays as {' and '.join(plays_as)} for {' and '.join(all_teams)}")
    elif plays_as:
        rel_parts.append(f"plays as {' and '.join(plays_as)}")
    elif captains_for and plays_for:
        if len(all_teams) >= 2:
            rel_parts.append(f"captains both {' and '.join(all_teams)}")
        else:
            rel_parts.append(f"captains {' and '.join(all_teams)}")
    elif captains_for:
        if len(captains_for) >= 2:
            rel_parts.append(f"captains both {' and '.join(captains_for)}")
        else:
            rel_parts.append(f"captains {' and '.join(captains_for)}")
    elif plays_for:
        rel_parts.append(f"plays for {' and '.join(plays_for)}")


    if other_clauses:
        rel_parts.extend(other_clauses)

    if rel_parts:
        if len(rel_parts) == 1:
            parts.append("who " + rel_parts[0])
        else:
            first_clause = rel_parts[0]
            rest_clauses = rel_parts[1:]
            parts.append("who " + first_clause + " and " + " and ".join(rest_clauses))

    fluid = " ".join(parts) + "."
    return [fluid]






def build_explanation(claims: list[ClaimExplainInput]) -> ExplainabilityResult:
    # Sort claims deterministically by text to guarantee ordering invariance across async runs
    sorted_claims = sorted(claims, key=lambda c: (c.text.strip().lower(), str(c.verdict)))

    supported = [c for c in sorted_claims if c.verdict == ClaimVerdict.SUPPORTED]
    contradicted = [c for c in sorted_claims if c.verdict == ClaimVerdict.CONTRADICTED]
    insufficient = [c for c in sorted_claims if c.verdict == ClaimVerdict.INSUFFICIENT]

    # Build rich fluid verified answer
    supported_texts = [c.text for c in supported]
    verified_parts = _synthesize_supported_claims(supported_texts)


    if contradicted:
        for c in contradicted:
            claim_text = c.text.rstrip(".!?")
            verified_parts.append(
                f"Note: The claim that '{claim_text}' is contradicted by retrieved evidence."
            )

    if not verified_parts:
        if insufficient:
            verified_answer = "Verification incomplete: Insufficient evidence available to confirm response claims."
        else:
            verified_answer = "No factual claims verified."
    else:
        ans = " ".join(verified_parts)
        verified_answer = f"Verified: {ans}" if not ans.startswith("Verified:") else ans


    lines = [
        f"{len(claims)} claim(s) checked: "
        f"{len(supported)} supported, {len(contradicted)} contradicted, "
        f"{len(insufficient)} insufficient evidence."
    ]
    if supported:
        lines.append("Verified claims:")
        for c in supported:
            lines.append(f"  - \"{c.text}\" (supported)")
    if contradicted:
        lines.append("Contradicted claims:")
        for c in contradicted:
            lines.append(f"  - \"{c.text}\" (contradicted)")
    if insufficient:
        lines.append("Claims with insufficient evidence:")
        for c in insufficient:
            lines.append(f"  - \"{c.text}\"")

    fabricated = [c for c in claims if c.fabricated_alternative]
    if fabricated:
        lines.append(
            "Likely fabricated alternatives (introduced after the response "
            "denied knowing the actual entity asked about):"
        )
        for c in fabricated:
            lines.append(f"  - \"{c.text}\"")

    explanation = "\n".join(lines)

    contradictions = [
        {
            "claim": c.text,
            "confidence": c.confidence,
            "evidence": c.evidence,
        }
        for c in contradicted
    ]

    return ExplainabilityResult(
        verified_answer=verified_answer,
        explanation=explanation,
        contradictions=contradictions,
    )
