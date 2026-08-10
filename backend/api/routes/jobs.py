"""
Phase 1 job endpoints - prove out DB + DI + UUID generation end to end.

These are intentionally minimal. Phase 11 formalizes the final public API
(POST /api/analyze, GET /api/job, GET /api/result, GET /api/history) on
top of the same JobService once the pipeline (Phase 4+) exists.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import AppError
from database.base import get_db
from models.schemas import JobCreateRequest, JobResponse
from services.job_service import JobService

router = APIRouter(prefix="/api", tags=["jobs"])


def get_job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    payload: JobCreateRequest,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    job = await service.create_job(query=payload.query)
    return JobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    try:
        job = await service.get_job(job_id)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    return JobResponse.model_validate(job)


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    service: JobService = Depends(get_job_service),
) -> list[JobResponse]:
    jobs = await service.list_jobs(limit=limit, offset=offset)
    return [JobResponse.model_validate(j) for j in jobs]
