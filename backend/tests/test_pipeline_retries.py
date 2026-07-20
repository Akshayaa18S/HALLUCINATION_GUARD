import asyncio
import pytest
import sys
import os

# Ensure repository root is on sys.path so `backend` package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.services.pipeline_service import PipelineService


class FakeDB:
    def add(self, obj):
        return None

    def commit(self):
        return None

    def refresh(self, obj):
        return None


class DummyJob:
    def __init__(self, job_id: str, input_text: str = None):
        self.job_id = job_id
        self.input_text = input_text
        self.input_image_path = None


@pytest.mark.asyncio
async def test_run_stage_retries_succeeds_after_retries():
    db = FakeDB()
    pipeline = PipelineService(db, dev_mode=False)
    # allow up to 3 retries
    pipeline.max_stage_retries = 3

    job = DummyJob("test-job-1", input_text="hello")

    # handler that fails twice then succeeds
    calls = {"count": 0}

    async def flaky_handler(state):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient error")
        return {"flaky_result": True}

    pipeline_state = {"job_id": job.job_id, "input_text": job.input_text}

    # should not raise
    await pipeline._run_stage(job, 99, "flaky", 50, None, pipeline_state, flaky_handler)

    assert pipeline_state.get("flaky_result") is True
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_run_stage_retries_fails_when_exceeds():
    db = FakeDB()
    pipeline = PipelineService(db, dev_mode=False)
    pipeline.max_stage_retries = 2

    job = DummyJob("test-job-2", input_text="hi")

    async def always_fail(state):
        raise RuntimeError("permanent error")

    pipeline_state = {"job_id": job.job_id, "input_text": job.input_text}

    with pytest.raises(RuntimeError):
        await pipeline._run_stage(job, 100, "always_fail", 10, None, pipeline_state, always_fail)
