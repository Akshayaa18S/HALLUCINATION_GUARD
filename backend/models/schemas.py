"""Pydantic schemas (API-facing shapes). Kept separate from the ORM
models in database/models.py so the API contract can evolve
independently of the storage layer."""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from models.enums import JobStatus

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class SignupRequest(BaseModel):
    email: str = Field(..., description="A real, unique email address")
    password: str = Field(..., min_length=8, description="At least 8 characters")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class JobCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user query/prompt to analyze")


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    query: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The prompt to generate and fact-check")


class AnalyzeAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class ClaimResponse(BaseModel):
    text: str
    entities: list[dict] = Field(default_factory=list)
    verdict: str
    confidence: float | None = None
    evidence: list[dict] = Field(default_factory=list)

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_claim(cls, claim) -> "ClaimResponse":
        return cls(
            text=claim.text,
            entities=claim.get_entities(),
            verdict=claim.verdict,
            confidence=claim.confidence,
            evidence=claim.get_evidence(),
        )


class ResultResponse(BaseModel):
    job_id: str
    generated_response: str | None = None
    verified_answer: str | None = None
    explanation: str | None = None
    overall_confidence: float | None = None
    hallucination_score: float | None = None
    processing_time_ms: float | None = None
    claims: list[ClaimResponse] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def from_orm_result(cls, result) -> "ResultResponse":
        return cls(
            job_id=result.job_id,
            generated_response=result.generated_response,
            verified_answer=result.verified_answer,
            explanation=result.explanation,
            overall_confidence=result.overall_confidence,
            hallucination_score=result.hallucination_score,
            processing_time_ms=result.processing_time_ms,
            claims=[ClaimResponse.from_orm_claim(c) for c in result.claims],
            created_at=result.created_at,
        )


class HistoryItem(BaseModel):
    job_id: str
    query: str | None = None
    hallucination_score: float | None = None
    overall_confidence: float | None = None
    created_at: datetime


class StageResponse(BaseModel):
    stage_name: str
    status: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_stage(cls, stage) -> "StageResponse":
        return cls(
            stage_name=stage.stage_name,
            status=stage.status,
            started_at=stage.started_at,
            ended_at=stage.ended_at,
            duration_ms=stage.duration_ms,
            error_message=stage.error_message,
            metadata=stage.get_metadata(),
        )
