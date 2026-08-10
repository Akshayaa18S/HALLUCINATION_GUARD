"""
ORM models.

Phase 1 owns Job. Phase 2 adds:
  - PipelineStage: one row per stage execution for a job (timing/status)
  - Result: one row per completed job (final verified answer + scores)
  - Claim: one row per atomic claim extracted from the response, with its
    verdict and evidence, linked to a Result.

All tables share the same `Base` / engine from database/base.py.
"""

import json
import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base
from models.enums import JobStatus, StageStatus, ClaimVerdict
from utils.time import utcnow


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return utcnow()


class User(Base):
    """A registered account, identified by email. Jobs are scoped to the
    user who created them so history/results stay private per-user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    jobs: Mapped[list["Job"]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.PENDING.value, nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped["User"] = relationship(back_populates="jobs")
    stages: Mapped[list["PipelineStage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="PipelineStage.started_at"
    )
    result: Mapped["Result"] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status}>"


class PipelineStage(Base):
    """One row per stage execution. Written by execution/manager.py so every
    stage's start/end/status/duration is auditable after the fact."""

    __tablename__ = "pipeline_stages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    stage_name: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=StageStatus.PENDING.value)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=True)

    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    stage_metadata: Mapped[str] = mapped_column(Text, nullable=True)  # JSON-encoded

    job: Mapped["Job"] = relationship(back_populates="stages")

    def get_metadata(self) -> dict:
        return json.loads(self.stage_metadata) if self.stage_metadata else {}

    def set_metadata(self, data: dict) -> None:
        self.stage_metadata = json.dumps(data, default=str)

    def __repr__(self) -> str:
        return f"<PipelineStage job={self.job_id} stage={self.stage_name} status={self.status}>"


class Result(Base):
    """Final output of a completed job."""

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id"), nullable=False, unique=True, index=True
    )

    generated_response: Mapped[str] = mapped_column(Text, nullable=True)
    verified_answer: Mapped[str] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)

    overall_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    hallucination_score: Mapped[float] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped["Job"] = relationship(back_populates="result")
    claims: Mapped[list["Claim"]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Result job={self.job_id} hallucination_score={self.hallucination_score}>"


class Claim(Base):
    """One atomic factual claim extracted from the generated response."""

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    result_id: Mapped[str] = mapped_column(ForeignKey("results.id"), nullable=False, index=True)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    entities: Mapped[str] = mapped_column(Text, nullable=True)  # JSON-encoded list
    verdict: Mapped[str] = mapped_column(
        String(20), default=ClaimVerdict.INSUFFICIENT.value, nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    evidence: Mapped[str] = mapped_column(Text, nullable=True)  # JSON-encoded list

    result: Mapped["Result"] = relationship(back_populates="claims")

    def get_entities(self) -> list:
        return json.loads(self.entities) if self.entities else []

    def set_entities(self, data: list) -> None:
        self.entities = json.dumps(data, default=str)

    def get_evidence(self) -> list:
        return json.loads(self.evidence) if self.evidence else []

    def set_evidence(self, data: list) -> None:
        self.evidence = json.dumps(data, default=str)

    def __repr__(self) -> str:
        return f"<Claim result={self.result_id} verdict={self.verdict}>"
