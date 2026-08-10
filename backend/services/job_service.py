"""
Service layer for job management.

This is deliberately thin in Phase 1 (create / get / list / update status).
Phase 4's Pipeline Engine will call `update_status` and eventually attach
results (added in Phase 2) as each stage completes.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import JobNotFoundError
from database.models import Job
from models.enums import JobStatus

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(self, query: str, user_id: str | None = None) -> Job:
        job = Job(query=query, status=JobStatus.PENDING.value, user_id=user_id)
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        logger.info("Created job %s (user=%s)", job.id, user_id)
        return job

    async def get_job(self, job_id: str) -> Job:
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> list[Job]:
        result = await self.db.execute(
            select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(
        self, job_id: str, status: JobStatus, error_message: str | None = None
    ) -> Job:
        job = await self.get_job(job_id)
        job.status = status.value
        if error_message is not None:
            job.error_message = error_message
        await self.db.commit()
        await self.db.refresh(job)
        logger.info("Job %s -> %s", job_id, status.value)
        return job
