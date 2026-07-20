import asyncio
import pytest

from datetime import datetime

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.services.pipeline_service import PipelineService
from backend.services.job_manager import JobManager
from backend.models.job import Job


class FlakyPipeline(PipelineService):
    """Pipeline with a flaky stage that fails n-1 times then succeeds."""
    def __init__(self, db, fail_times=2, *args, **kwargs):
        super().__init__(db, *args, **kwargs)
        self.fail_times = fail_times

    async def _stage_4(self, state):
        attempts = state.get("_stage_4_attempts", 0)
        attempts += 1
        state["_stage_4_attempts"] = attempts
        if attempts <= self.fail_times:
            raise RuntimeError("simulated flaky failure")
        return await super()._stage_4(state)


@pytest.fixture(scope="module")
def db():
    init_db()
    db = SessionLocal()
    yield db
    db.close()


@pytest.mark.asyncio
async def test_stage_retries(db):
    # ensure retries is 3 for test
    settings.JOB_MAX_RETRIES = 3

    # create a dummy job
    job = JobManager.create_job(db, type("R", (), {"input_text": "Paris capital"}))

    pipeline = FlakyPipeline(db, fail_times=2)

    events = []

    async def cb(payload):
        events.append(payload)

    result = await pipeline.execute(job.job_id, progress_callback=cb)

    assert result["completed"] is True
    # ensure that the flaky stage eventually succeeded
    assert db.query(Job).filter(Job.job_id == job.job_id).first().status == "completed"
    # check that we received final event
    assert any(e.get("type") == "final" for e in events)
