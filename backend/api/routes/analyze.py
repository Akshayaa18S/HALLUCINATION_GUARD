"""
Phase 11 - public API.

POST /api/analyze  - kicks off the full pipeline, returns immediately with a job_id
GET  /api/job/{id}    - poll job status
GET  /api/result/{id} - fetch the final result once the job is completed
GET  /api/history     - list past jobs with their scores

The pipeline runs in a FastAPI BackgroundTask with its OWN database
session (background tasks outlive the request's session, which gets
closed as soon as the response is sent).
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from core.exceptions import AppError, ForbiddenError
from database.base import AsyncSessionLocal, get_db
from database.models import User
from execution.manager import PipelineManager
from models.enums import JobStatus
from models.schemas import (
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    HistoryItem,
    JobResponse,
    ResultResponse,
    StageResponse,
)
from services.job_service import JobService
from services.result_service import ResultRepository
from services.stage_service import StageRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])


async def _run_pipeline_background(job_id: str, query: str) -> None:
    """Runs in its own session/connection, independent of the request's."""
    async with AsyncSessionLocal() as db:
        manager = PipelineManager(db)
        try:
            await manager.run_job(job_id, query)
        except Exception:
            logger.exception("Pipeline run failed for job %s", job_id)


_predictor_instance = None


def _get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        from predict import MultiHaluDetPredictor
        _predictor_instance = MultiHaluDetPredictor()
    return _predictor_instance


@router.post("/predict")
async def predict_direct(payload: AnalyzeRequest):
    """Direct, synchronous MultiHaluDet model inference endpoint."""
    predictor = _get_predictor()
    return predictor.predict(payload.query)


@router.post("/analyze", response_model=AnalyzeAcceptedResponse, status_code=202)
async def analyze(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyzeAcceptedResponse:
    job_service = JobService(db)
    job = await job_service.create_job(query=payload.query, user_id=current_user.id)
    background_tasks.add_task(_run_pipeline_background, job.id, payload.query)
    return AnalyzeAcceptedResponse(job_id=job.id, status=JobStatus(job.status))


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    try:
        job = await JobService(db).get_job(job_id)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    if job.user_id and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to access this job")
    return JobResponse.model_validate(job)


@router.get("/job/{job_id}/stages", response_model=list[StageResponse])
async def get_job_stages(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[StageResponse]:
    try:
        job = await JobService(db).get_job(job_id)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    if job.user_id and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not allowed to access this job")
    stages = await StageRepository(db).list_for_job(job_id)
    return [StageResponse.from_orm_stage(s) for s in stages]


@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ResultResponse:
    try:
        job = await JobService(db).get_job(job_id)
        if job.user_id and job.user_id != current_user.id:
            raise ForbiddenError("Not allowed to access this result")
        result = await ResultRepository(db).get_by_job_id(job_id)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    return ResultResponse.from_orm_result(result)


@router.get("/history", response_model=list[HistoryItem])
async def get_history(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[HistoryItem]:
    results = await ResultRepository(db).list_recent(
        limit=limit, offset=offset, user_id=current_user.id
    )
    return [
        HistoryItem(
            job_id=r.job_id,
            query=r.job.query if r.job else None,
            hallucination_score=r.hallucination_score,
            overall_confidence=r.overall_confidence,
            created_at=r.created_at,
        )
        for r in results
    ]
