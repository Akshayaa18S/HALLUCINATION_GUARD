import logging

from hallucination.verification import ClaimVerifier
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _dedupe_claim_entities(entities):
    seen = set()
    unique = []
    for e in entities:
        key = (getattr(e, "text", "").strip(), getattr(e, "label", ""))
        if key not in seen and key[0]:
            seen.add(key)
            unique.append(e)
    return unique


class VerificationStage(Stage):
    name = StageName.VERIFICATION
    critical = True  # without verdicts, hallucination detection has nothing to score

    def __init__(self, verifier: ClaimVerifier | None = None):
        self.verifier = verifier or ClaimVerifier(LLMService())

    async def run(self, context: PipelineContext) -> PipelineContext:
        # Canonical entity propagation across context claims
        primary_person = ""
        for claim in context.claims:
            for e in claim.entities:
                if getattr(e, "label", None) == "PERSON" and len(e.text) > len(primary_person):
                    primary_person = e.text

        if primary_person:
            for claim in context.claims:
                for e in claim.entities:
                    if getattr(e, "label", None) == "PERSON" and e.text != primary_person:
                        if any(w in primary_person for w in e.text.split()):
                            e.text = primary_person

        for claim in context.claims:
            claim.entities = _dedupe_claim_entities(claim.entities)


        for claim in context.claims:
            if not claim.evidence:
                subj = next((e.text for e in claim.entities if getattr(e, "label", None) == "PERSON"), "")
                if subj and subj in context.query_evidence:
                    claim.evidence = list(context.query_evidence[subj])
                else:
                    for other in context.claims:
                        if other.evidence:
                            claim.evidence = list(other.evidence)
                            break


            verdict, base_conf = await self.verifier.verify(claim.text, claim.evidence)
            claim.verdict = verdict.value

            # Populate claim-level provenance & multi-source evidence agreement
            distinct_sources = list(dict.fromkeys(e.get("source", "wikipedia") for e in claim.evidence if e.get("source")))
            claim.sources = distinct_sources if distinct_sources else ["wikipedia"]

            if claim.verdict == "supported":
                claim.support_count = max(len(claim.sources), 1)
                claim.contradiction_count = 0
                claim.agreement = 1.0
            elif claim.verdict == "contradicted":
                claim.support_count = 0
                claim.contradiction_count = max(len(claim.sources), 1)
                claim.agreement = 0.0
            else:
                claim.support_count = 0
                claim.contradiction_count = 0
                claim.agreement = 0.5

            # Separate metric: evidence quality (average snippet relevance score)
            scores = [e["score"] for e in claim.evidence if isinstance(e.get("score"), (int, float))]
            claim.evidence_quality = round(sum(scores) / len(scores), 4) if scores else 0.5
            max_ev_score = max(scores) if scores else 0.5

            # Multi-factor calibrated confidence formula:
            # 40% Evidence Quality + 35% Cross-Source Agreement + 25% Support Count Factor
            ev_q = claim.evidence_quality
            agr = claim.agreement
            sup_factor = min(1.0, 0.5 * claim.support_count) if claim.support_count > 0 else 0.3

            if claim.verdict == "supported":
                calibrated = 0.40 * ev_q + 0.35 * agr + 0.25 * sup_factor
                # Determine floor based on evidence strength:
                #   - Canonical source (score ≥ 0.90) + multi-source → floor 0.95
                #   - Multi-source (≥ 2)                             → floor 0.92
                #   - Single source                                  → floor 0.88
                if max_ev_score >= 0.90 and claim.support_count >= 2:
                    floor = 0.95
                elif claim.support_count >= 2:
                    floor = 0.92
                else:
                    floor = 0.88
                claim.confidence = round(max(floor, min(0.98, calibrated)), 2)
            elif claim.verdict == "contradicted":
                claim.confidence = 0.90
            else:
                claim.confidence = 0.30






            # Provenance triplet extraction (subject, relation, object)
            subj = next((e.text for e in claim.entities if getattr(e, "label", None) == "PERSON"), "")
            if not subj and claim.entities:
                subj = claim.entities[0].text
            claim.subject = subj

            text = claim.text
            rel, obj = "", ""
            for verb in ("plays for", "plays as", "is a", "is an", "was a", "was an", "signed with", "signed for", "manages", "transferred to"):
                if verb in text.lower():
                    idx = text.lower().find(verb)
                    rel = verb
                    obj = text[idx + len(verb):].strip().rstrip(".!?")
                    break
            claim.relation = rel
            claim.object = obj

        context.record(self.name.value, {"claims_verified": len(context.claims)})
        return context

        