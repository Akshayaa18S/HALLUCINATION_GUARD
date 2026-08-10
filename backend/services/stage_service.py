"""Repository for pipeline stage timing/status records."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import PipelineStage
from models.enums import StageStatus
from utils.time import utcnow

logger = logging.getLogger(__name__)


class StageRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def start(self, job_id: str, stage_name: str) -> PipelineStage:
        stage = PipelineStage(
            job_id=job_id,
            stage_name=stage_name,
            status=StageStatus.RUNNING.value,
            started_at=utcnow(),
        )
        self.db.add(stage)
        await self.db.commit()
        await self.db.refresh(stage)
        return stage

    async def finish(
        self,
        stage: PipelineStage,
        status: StageStatus,
        metadata: dict | None = None,
        error_message: str | None = None,
    ) -> PipelineStage:
        # Both sides are naive UTC (see utils/time.py) so this subtraction is
        # always safe, regardless of whether stage.started_at is the
        # in-memory value or one that's round-tripped through SQLite (which
        # would otherwise strip tzinfo and cause a naive/aware mismatch).
        stage.ended_at = utcnow()
        if stage.started_at is not None:
            delta = stage.ended_at - stage.started_at
            stage.duration_ms = delta.total_seconds() * 1000
        stage.status = status.value
        stage.error_message = error_message
        if metadata:
            stage.set_metadata(metadata)
        await self.db.commit()
        await self.db.refresh(stage)
        logger.info(
            "Stage %s (%s) -> %s in %.1fms",
            stage.stage_name, stage.job_id, status.value, stage.duration_ms or -1,
        )
        return stage

    async def list_for_job(self, job_id: str) -> list[PipelineStage]:
        res = await self.db.execute(
            select(PipelineStage)
            .where(PipelineStage.job_id == job_id)
            .order_by(PipelineStage.started_at)
        )
        return list(res.scalars().all())
