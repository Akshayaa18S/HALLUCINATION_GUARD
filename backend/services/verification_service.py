"""Evidence-based verification and final claim-decision aggregation."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.ollama_service import OllamaService, OllamaServiceError
from services.knowledge_base import knowledge_base
from services.retriever import RetrievalDocument

logger = logging.getLogger("hallucination_guard.verification_service")
RELEVANCE_THRESHOLD = 0.35
MAX_CLAIMS = 5
_CONVERSATIONAL = re.compile(
    r"^\s*(?:i(?:'m| am)\s+afraid\s+that(?:'s| is)?\s+not\s+correct|that(?:'s| is)\s+not\s+accurate|that(?:'s| is)\s+incorrect|i\s+think|in\s+my\s+opinion|however|therefore|overall|it(?:'s| is)\s+worth\s+noting|if\s+you\s+have\s+any\s+questions|is\s+there\s+anything\s+else\s+i\s+can\s+help\s+clarify|would\s+you\s+like\s+more\s+information|i(?:'d| would)\s+be\s+happy\s+to\s+help|feel\s+free\s+to\s+ask)\b",
    re.IGNORECASE,
)
_TRUSTED_SOURCES = {"wikipedia", "fever", "local kb", "knowledge base", "halueval"}

class VerificationService:
    """Verifies factual propositions and aggregates three exclusive states."""

    def __init__(self, ollama_service: Optional[OllamaService] = None, retriever=None) -> None:
        self.ollama_service = ollama_service or OllamaService()
        self.retriever = retriever or knowledge_base

    def verify(self, query: str, generated_response: str, top_k: int = 5) -> Dict[str, Any]:
        claims = self._split_claims(generated_response)
        if not claims:
            return self._empty_result()
        claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]] = []
        for claim in claims:
            docs = [(doc, score) for doc, score in self.retriever.retrieve(claim, k=top_k) if (doc.text or "").strip() and doc.source.casefold() != "halueval"]
            claim_evidence.append((claim, docs))
        if not any(docs for _, docs in claim_evidence):
            # Absence of evidence is an explicit verification state, never a hallucination.
            return self._aggregate("{}", claim_evidence, query)
        prompt = self._build_prompt(query, claim_evidence)
        system_prompt = (
            "You are a strict evidence-based verifier. For EACH claim, choose exactly one state: "
            "supported, contradicted, or insufficient evidence, using only the listed evidence. "
            "A contradiction requires explicit conflicting evidence. Return JSON only: "
            '{"verified_answer":"...", "claims":[{"claim":"...", "supported":true/false, '
            '"contradicted":true/false, "explanation":"..."}]}'
        )
        try:
            raw = self.ollama_service.generate(prompt, system_prompt=system_prompt, format_json=True)
        except OllamaServiceError as exc:
            logger.warning("Ollama verification failed; marking claims insufficient: %s", exc)
            raw = "{}"
        return self._aggregate(raw, claim_evidence, query)

    def _split_claims(self, text: str) -> List[str]:
        """Keep factual propositions only and resolve local pronoun subjects."""
        sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        claims: List[str] = []
        last_subject = ""
        for sentence in sentences:
            candidate = sentence.strip()
            if len(candidate.split()) < 3 or _CONVERSATIONAL.match(candidate):
                logger.info("FEVER claim skipped: original=%r reason=conversational_or_nonfactual", candidate)
                continue
            # Remove hedging/adjectival fillers without changing the proposition.
            candidate = re.sub(r"\b(?:actually|separate)\s+", "", candidate, flags=re.IGNORECASE)
            subject_match = re.match(r"^([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})\s+(?:is|are|was|were|has|have|had|located)\b", candidate)
            if re.match(r"^It\s+", candidate, re.IGNORECASE):
                if not last_subject:
                    logger.info("FEVER claim skipped: original=%r reason=unresolved_pronoun", candidate)
                    continue
                candidate = re.sub(r"^It\b", last_subject, candidate, flags=re.IGNORECASE)
            elif subject_match:
                last_subject = subject_match.group(1)
            if len(candidate.split()) >= 3:
                claims.append(candidate)
        return claims[:MAX_CLAIMS]

    def _build_prompt(self, query: str, claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]]) -> str:
        # Deliberately omit the user assertion: this verifier assesses only
        # the model-generated claims against retrieved evidence.
        blocks = []
        for index, (claim, docs) in enumerate(claim_evidence, start=1):
            blocks.append(f"Claim {index}: {claim}")
            blocks.extend(f"  Evidence [{doc.source}] (similarity={score:.2f}): {doc.text}" for doc, score in docs)
            if not docs:
                blocks.append("  Evidence: (none retrieved)")
        blocks.append("\nReturn JSON only, verifying every claim listed above.")
        return "\n".join(blocks)

    def _aggregate(self, raw_text: str, claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]], user_claim: str = "") -> Dict[str, Any]:
        try:
            data = json.loads(raw_text)
        except (TypeError, json.JSONDecodeError):
            data = {}
        llm_claims = {str(item.get("claim", "")).strip(): item for item in data.get("claims", []) if isinstance(item, dict)}
        supporting_documents: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, Any]] = []
        claim_verdicts: List[Dict[str, str]] = []
        seen, similarities = set(), []
        counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT_EVIDENCE": 0}
        for claim, docs in claim_evidence:
            verdict = llm_claims.get(claim, {})
            relevant = [(doc, score) for doc, score in docs if score >= RELEVANCE_THRESHOLD]
            similarities.append(max((score for _, score in docs), default=0.0))
            trusted_support = [(doc, score) for doc, score in relevant if doc.source.casefold() in _TRUSTED_SOURCES]
            # Each trusted source can independently support a factual claim.
            # Missing FEVER evidence never negates relevant Wikipedia/local KB evidence.
            # A retained trusted source is independent positive evidence for
            # this generated claim; it cannot be turned into a contradiction by
            # the user's original assertion or an inconsistent model flag.
            if trusted_support or (bool(verdict.get("supported")) and relevant):
                state = "SUPPORTED"
                target = supporting_documents
            elif bool(verdict.get("contradicted")) and relevant:
                state = "CONTRADICTED"
                target = contradictions
            else:
                state = "INSUFFICIENT_EVIDENCE"
                target = None
            counts[state] += 1
            claim_verdicts.append({"claim": claim, "state": state, "explanation": str(verdict.get("explanation", ""))})
            if target is not None:
                for doc, score in relevant[:2]:
                    key = (doc.source, doc.text.strip().lower())
                    if key not in seen:
                        entry = {"source": doc.source, "content": doc.text, "score": round(score, 4), "claim": claim}
                        target.append(entry)
                        seen.add(key)
        total = len(claim_evidence)
        # Final consistency invariant: a "yes" verdict requires retained
        # contradictory evidence; retained support with no contradictions is no.
        if contradictions:
            decision = "yes"
            message = "The following claims contradict retrieved evidence."
        elif supporting_documents:
            decision = "no"
            message = "The generated response is supported by retrieved evidence."
        else:
            decision = "uncertain"
            message = "There was insufficient evidence to verify one or more claims."
        # Combine evidence quality, resolved-claim coverage, and independent-source agreement.
        resolved_scores = [entry["score"] for entry in supporting_documents + contradictions if isinstance(entry.get("score"), (int, float))]
        quality = sum(resolved_scores) / len(resolved_scores) if resolved_scores else 0.0
        coverage = (counts["SUPPORTED"] + counts["CONTRADICTED"]) / total if total else 0.0
        source_count = len({entry.get("source") for entry in supporting_documents + contradictions if entry.get("source")})
        agreement = min(1.0, source_count / 2.0)
        confidence = quality * (0.60 + 0.25 * coverage + 0.15 * agreement)
        verified_answer = self._verified_answer(user_claim, decision, claim_verdicts, supporting_documents, contradictions)
        return {
            "is_hallucination": decision,
            "final_decision": decision,
            "confidence": round(confidence, 4),
            "verified_answer": verified_answer,
            "evidence": supporting_documents,
            "contradictions": contradictions,
            "supporting_documents": supporting_documents,
            "explanation": message,
            "claim_verdicts": claim_verdicts,
            "claim_count": total,
            "supported_count": counts["SUPPORTED"],
            "contradicted_count": counts["CONTRADICTED"],
            "insufficient_count": counts["INSUFFICIENT_EVIDENCE"],
            "unsupported_claim_count": counts["INSUFFICIENT_EVIDENCE"],
            "mean_similarity": round(float(np.mean(similarities)) if similarities else 0.0, 4),
        }

    @staticmethod
    def _verified_answer(user_claim: str, decision: str, claim_verdicts: List[Dict[str, str]], supporting_documents: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> str:
        """Assess the user's claim while the decision remains about model output."""
        supported_claims = [item["claim"] for item in claim_verdicts if item.get("state") == "SUPPORTED" and item.get("claim")]
        factual_summary = " ".join(supported_claims)
        if decision == "no":
            if VerificationService._user_claim_conflicts_with_supported_fact(user_claim, factual_summary):
                return f"The user's claim is false. {factual_summary}".strip()
            if factual_summary:
                return f"The user's claim is supported by retrieved evidence. {factual_summary}"
            return "The user's claim is supported by retrieved evidence."
        if decision == "yes":
            detail = str(contradictions[0].get("content", "")) if contradictions else ""
            return f"The generated response is contradicted by evidence. {detail}".strip()
        return "There is insufficient evidence to verify this claim."

    @staticmethod
    def _user_claim_conflicts_with_supported_fact(user_claim: str, factual_summary: str) -> bool:
        """Detect straightforward corrections without using user-claim truth as a hallucination signal."""
        user = (user_claim or "").casefold()
        fact = (factual_summary or "").casefold()
        if not user or not fact:
            return False
        if (" not " in user) != (" not " in fact):
            return True
        regions = {"african", "africa", "asian", "asia", "european", "europe", "south asia"}
        user_regions = {term for term in regions if term in user}
        fact_regions = {term for term in regions if term in fact}
        return bool(user_regions and fact_regions and user_regions.isdisjoint(fact_regions))


    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "is_hallucination": "uncertain", "final_decision": "uncertain", "confidence": 0.0,
            "verified_answer": "There were no factual claims to verify.", "evidence": [], "contradictions": [],
            "supporting_documents": [], "explanation": "There was insufficient evidence to verify one or more claims.",
            "claim_verdicts": [], "claim_count": 0, "supported_count": 0, "contradicted_count": 0,
            "insufficient_count": 0, "unsupported_claim_count": 0, "mean_similarity": 0.0,
        }

verification_service = VerificationService()
