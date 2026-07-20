import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from backend.models.job import Job, JobStatus
from backend.services.job_manager import JobManager


def test_cancel_job_sets_cancelled_status(db_session):
    job = Job(
        job_id="job-cancel-test",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="Request cancellation",
    )
    db_session.add(job)
    db_session.commit()

    cancelled = JobManager.cancel_job(db_session, job.job_id)
    assert cancelled.status == JobStatus.CANCELLED.value
    assert cancelled.completed_at is not None


def test_get_job_status_returns_current_stage_info(db_session):
    job = Job(
        job_id="job-status-test",
        status=JobStatus.RUNNING.value,
        input_type="text",
        input_text="Status query",
    )
    db_session.add(job)
    db_session.commit()

    status = JobManager.get_job_status(db_session, job.job_id)
    assert status.job_id == job.job_id
    assert status.status == JobStatus.RUNNING.value
    assert status.input_type == "text"
