"""Shared enums. Kept separate from models/schemas so both can import it
without a circular dependency."""

import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ClaimVerdict(str, enum.Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    NOT_VERIFIABLE = "not_verifiable"



class StageName(str, enum.Enum):
    QUERY_GROUNDING = "query_grounding"
    GENERATION = "generation"
    CLAIM_EXTRACTION = "claim_extraction"
    COREFERENCE_RESOLUTION = "coreference_resolution"
    ENTITY_EXTRACTION = "entity_extraction"
    WIKIPEDIA_RETRIEVAL = "wikipedia_retrieval"
    FEVER_RETRIEVAL = "fever_retrieval"
    EVIDENCE_RANKING = "evidence_ranking"
    VERIFICATION = "verification"
    QUERY_CONSISTENCY = "query_consistency"
    HALLUCINATION_DETECTION = "hallucination_detection"
    EXPLAINABILITY = "explainability"