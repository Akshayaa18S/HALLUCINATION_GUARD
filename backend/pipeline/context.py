"""
The object every stage reads from and writes to. Keeping this as one
dataclass (rather than passing raw dicts around) makes it obvious what
each stage is allowed to touch, and makes Phase 9's rule ("hallucination
detection only ever looks at claims, never the user prompt") easy to
enforce - the detector simply isn't given `query`.
"""

from dataclasses import dataclass, field
from typing import Any


from hallucination.verifiability import ClaimType


@dataclass
class Entity:
    text: str
    label: str  # PERSON, LOCATION, ORGANIZATION, COUNTRY, CITY, DATE, NUMBER, SPORTS_TEAM, EVENT, PRODUCT
    start: int = -1
    end: int = -1


@dataclass
class ClaimContext:
    text: str
    entities: list[Entity] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # [{source, text, url, score}]
    verdict: str | None = None       # ClaimVerdict value
    confidence: float | None = None
    fabricated_alternative: bool = False

    # Claim verifiability classification
    claim_type: ClaimType = ClaimType.OBJECTIVE_FACT

    # Claim-level provenance & multi-source evidence agreement
    subject: str = ""
    relation: str = ""
    object: str = ""
    sources: list[str] = field(default_factory=list)
    support_count: int = 0
    contradiction_count: int = 0
    agreement: float = 1.0
    evidence_quality: float = 0.0


    @property
    def is_verifiable(self) -> bool:
        return self.claim_type in {
            ClaimType.OBJECTIVE_FACT,
            ClaimType.OBJECTIVE_NEGATIVE_FACT,
            ClaimType.UNCERTAIN_FACT,
        }




@dataclass
class PipelineContext:
    job_id: str
    query: str

    # Populated by pipeline.stages.query_grounding, which runs *before*
    # generation: entities found in the user's own query, and whatever
    # Wikipedia evidence exists for them. knowledge_context is the flattened
    # text handed to the LLM as reference material during generation, and
    # query_evidence is kept structured so later stages (query_consistency)
    # can check the response against it without re-retrieving.
    query_entities: list[Entity] = field(default_factory=list)
    query_evidence: dict[str, list[dict]] = field(default_factory=dict)
    knowledge_context: str = ""

    generated_response: str = ""
    claims: list[ClaimContext] = field(default_factory=list)

    verified_answer: str = ""
    explanation: str = ""
    contradictions: list[dict] = field(default_factory=list)

    hallucination_score: float | None = None
    overall_confidence: float | None = None

    # per-stage scratch space, e.g. {"wikipedia_retrieval": {"pages_hit": 3}}
    stage_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, stage_name: str, metadata: dict[str, Any]) -> None:
        self.stage_metadata[stage_name] = metadata
