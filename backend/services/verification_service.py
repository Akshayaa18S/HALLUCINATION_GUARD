"""Evidence-driven verification service for RAG-based hallucination detection.

Verifies the FULL generated response (not just its first sentence) by:
  1. splitting it into individual factual claims,
  2. retrieving evidence per claim and gating relevance with embedding similarity,
  3. asking the LLM to verify each claim using ONLY that evidence,
  4. discarding any "contradiction" whose evidence isn't actually topically
     related to the claim (a low-similarity match can't logically contradict
     anything - it's just an unrelated retrieval hit), and
  5. aggregating claim-level verdicts into one response-level result.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.ollama_service import OllamaService, OllamaServiceError
from services.knowledge_base import knowledge_base
from services.retriever import RetrievalDocument

logger = logging.getLogger("hallucination_guard.verification_service")

# Below this cosine similarity, evidence is considered topically unrelated to
# the claim and can never count as support OR contradiction.
RELEVANCE_THRESHOLD = 0.35
MAX_CLAIMS = 5


class VerificationService:
    """Uses retrieved evidence to verify a generated response with an LLM."""

    def __init__(self, ollama_service: Optional[OllamaService] = None, retriever=None) -> None:
        self.ollama_service = ollama_service or OllamaService()
        # `retriever` is injectable for tests; production code uses the
        # shared knowledge_base (static FEVER/HaluEval corpus + live Wikipedia).
        self.retriever = retriever or knowledge_base

    def verify(self, query: str, generated_response: str, top_k: int = 5) -> Dict[str, Any]:
        claims = self._split_claims(generated_response) or [generated_response.strip() or query]

        claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]] = []
        for claim in claims:
            docs = self.retriever.retrieve(claim, k=top_k)
            docs = [(doc, score) for doc, score in docs if (doc.text or "").strip()]
            claim_evidence.append((claim, docs))

        if not any(docs for _, docs in claim_evidence):
            raise OllamaServiceError("No evidence documents were available for verification.")

        prompt = self._build_prompt(query, claim_evidence)
        system_prompt = (
            "You are a strict evidence-based verifier. For EACH claim, decide if the provided "
            "evidence supports it, contradicts it, or is insufficient - using ONLY that evidence, "
            "never prior knowledge. Mark a claim as contradicted ONLY if the evidence explicitly "
            "states something that conflicts with the claim (e.g. a different value for the same "
            "fact). Evidence about a different topic than the claim is NOT a contradiction - treat "
            "it as insufficient evidence instead. If evidence is insufficient, say so explicitly. "
            "Return JSON only, no commentary, matching this shape:\n"
            '{"verified_answer": "...", "overall_confidence": 0-1, '
            '"claims": [{"claim": "...", "supported": true/false, "contradicted": true/false, '
            '"explanation": "..."}]}'
        )

        try:
            raw = self.ollama_service.generate(prompt, system_prompt=system_prompt, format_json=True)
        except OllamaServiceError as exc:
            logger.warning("Ollama verification failed: %s", exc)
            raise

        return self._aggregate(raw, claim_evidence)

    def _split_claims(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        return claims[:MAX_CLAIMS] if claims else ([text] if text else [])

    def _build_prompt(self, query: str, claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]]) -> str:
        blocks = [f"Original question: {query}\n"]
        for index, (claim, docs) in enumerate(claim_evidence, start=1):
            blocks.append(f"Claim {index}: {claim}")
            if not docs:
                blocks.append("  Evidence: (none retrieved)")
                continue
            for doc, score in docs:
                blocks.append(f"  Evidence [{doc.source}] (similarity={score:.2f}): {doc.text}")
        blocks.append("\nReturn JSON only, verifying every claim listed above.")
        return "\n".join(blocks)

    def _aggregate(
        self,
        raw_text: str,
        claim_evidence: List[Tuple[str, List[Tuple[RetrievalDocument, float]]]],
    ) -> Dict[str, Any]:
        import json

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = {}

        llm_claims = {
            str(item.get("claim", "")).strip(): item
            for item in (data.get("claims") or [])
            if isinstance(item, dict)
        }

        supporting_documents: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, Any]] = []
        seen_evidence_keys = set()
        unsupported_count = 0
        similarities: List[float] = []

        for claim, docs in claim_evidence:
            llm_verdict = llm_claims.get(claim, {})
            best_score = max((score for _, score in docs), default=0.0)
            similarities.append(best_score)

            relevant_docs = [(doc, score) for doc, score in docs if score >= RELEVANCE_THRESHOLD]
            supported = bool(llm_verdict.get("supported")) and bool(relevant_docs)
            # A contradiction requires BOTH the LLM saying so AND at least one
            # genuinely relevant piece of evidence - an unrelated retrieval
            # hit can never contradict the claim, regardless of what the LLM says.
            contradicted = bool(llm_verdict.get("contradicted")) and bool(relevant_docs)

            if not supported and not contradicted:
                unsupported_count += 1

            for doc, score in relevant_docs[:2]:
                key = (doc.source, doc.text.strip().lower())
                entry = {"source": doc.source, "content": doc.text, "score": round(score, 4)}
                if contradicted:
                    if key not in seen_evidence_keys:
                        contradictions.append({**entry, "claim": claim})
                elif supported and key not in seen_evidence_keys:
                    supporting_documents.append(entry)
                seen_evidence_keys.add(key)

        total_claims = max(1, len(claim_evidence))
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0

        overall_confidence = data.get("overall_confidence")
        try:
            confidence = float(overall_confidence)
            if confidence > 1:
                confidence = min(confidence / 100.0, 1.0)
        except (TypeError, ValueError):
            confidence = mean_similarity
        confidence = max(0.0, min(1.0, confidence))

        verified_answer = str(
            data.get("verified_answer")
            or "; ".join(
                llm_claims.get(c, {}).get("explanation", "")
                for c, _ in claim_evidence
                if llm_claims.get(c, {}).get("explanation")
            )
            or "Verification could not be completed from the retrieved evidence."
        )

        return {
            "is_hallucination": bool(contradictions) or unsupported_count == total_claims,
            "confidence": confidence,
            "verified_answer": verified_answer,
            "evidence": supporting_documents,
            "contradictions": contradictions,
            "supporting_documents": supporting_documents,
            "explanation": "",
            "claim_count": total_claims,
            "unsupported_claim_count": unsupported_count,
            "mean_similarity": round(mean_similarity, 4),
        }


verification_service = VerificationService()
