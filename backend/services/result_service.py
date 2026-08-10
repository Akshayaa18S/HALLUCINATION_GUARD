"""
Repository layer for Result + Claim persistence.

Kept separate from JobService (services/job_service.py) because results
are written once, at the end of a pipeline run, by execution/manager.py -
different lifecycle and different caller than job CRUD.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.exceptions import NotFoundError
from database.models import Claim, Job, Result

logger = logging.getLogger(__name__)


class ResultRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        job_id: str,
        generated_response: str,
        verified_answer: str,
        explanation: str,
        overall_confidence: float,
        hallucination_score: float,
        processing_time_ms: float,
        claims: list[dict],
    ) -> Result:
        result = Result(
            job_id=job_id,
            generated_response=generated_response,
            verified_answer=verified_answer,
            explanation=explanation,
            overall_confidence=overall_confidence,
            hallucination_score=hallucination_score,
            processing_time_ms=processing_time_ms,
        )
        for c in claims:
            claim = Claim(
                text=c["text"],
                verdict=c["verdict"],
                confidence=c.get("confidence"),
            )
            claim.set_entities(c.get("entities", []))
            claim.set_evidence(c.get("evidence", []))
            result.claims.append(claim)

        self.db.add(result)
        await self.db.commit()
        await self.db.refresh(result)
        logger.info("Persisted result for job %s (%d claims)", job_id, len(claims))
        return result

    async def get_by_job_id(self, job_id: str) -> Result:
        stmt = (
            select(Result)
            .where(Result.job_id == job_id)
            .options(selectinload(Result.claims))
        )
        res = await self.db.execute(stmt)
        result = res.scalar_one_or_none()
        if result is None:
            raise NotFoundError(f"No result for job '{job_id}'")
        return result

    async def list_recent(
        self, limit: int = 20, offset: int = 0, user_id: str | None = None
    ) -> list[Result]:
        stmt = (
            select(Result)
            .join(Result.job)
            .options(selectinload(Result.claims), selectinload(Result.job))
            .order_by(Result.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if user_id is not None:
            stmt = stmt.where(Job.user_id == user_id)
        res = await self.db.execute(stmt)
        return list(res.scalars().all())
