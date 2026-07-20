import os
import sys
import asyncio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from sqlalchemy.orm import sessionmaker

from backend.models.job import Job, JobStatus
from backend.models.stage import Stage
from backend.models.result import Result
from backend.models.base import Base
from backend.services.pipeline_service import PipelineService
from backend.services.job_manager import JobManager


@pytest.mark.asyncio
async def test_stage_functions_return_expected_outputs(db_session):
    job = Job(
        job_id="job-stage-test",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="What is the capital of Germany?",
    )
    db_session.add(job)
    db_session.commit()

    pipeline = PipelineService(db_session, dev_mode=False)
    state = {"job_id": job.job_id, "input_text": job.input_text}

    stage_1 = await pipeline._stage_1(state.copy())
    assert stage_1 == {
        "input_received": True,
        "input_type": "text",
        "user_query": "What is the capital of Germany?",
    }

    stage_2 = await pipeline._stage_2(state.copy())
    assert "generated_response" in stage_2
    assert "Germany" in stage_2["generated_response"]
    assert "capital" in stage_2["generated_response"].lower()

    stage_3 = await pipeline._stage_3(state.copy())
    assert "attention_maps" in stage_3 and isinstance(stage_3["attention_maps"], list)

    stage_4 = await pipeline._stage_4(state.copy())
    assert "multi_scale_attention" in stage_4 and isinstance(stage_4["multi_scale_attention"], list)

    stage_5 = await pipeline._stage_5(state.copy())
    assert stage_5["prediction"] in {True, False}
    assert 0 <= stage_5["probability"] <= 1
    assert 0 <= stage_5["confidence"] <= 100

    stage_6 = await pipeline._stage_6(state.copy())
    assert "supporting_documents" in stage_6
    assert "evidence" in stage_6

    stage_7 = await pipeline._stage_7(state.copy())
    assert "verified answer" in stage_7["explanation_text"].lower()
    assert "evidence" in stage_7["explanation_text"].lower()

    stage_8 = await pipeline._stage_8(state.copy())
    assert stage_8["analysis_completed"] is True
    assert isinstance(stage_8.get("execution_pipeline"), list)


@pytest.mark.asyncio
async def test_full_pipeline_execution_persists_result(db_session):
    job = Job(
        job_id="job-full-run",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="What is the capital of Germany?",
    )
    db_session.add(job)
    db_session.commit()

    pipeline = PipelineService(db_session, dev_mode=False)
    summary = await pipeline.execute(job.job_id)

    assert summary["completed"] is True
    assert summary["hallucination"] is False
    assert "processing_time" in summary

    persisted = db_session.query(Result).filter(Result.job_id == job.job_id).one_or_none()
    assert persisted is not None
    assert "Germany" in persisted.generated_response
    assert "Berlin" in persisted.verified_answer
    assert persisted.retrieved_evidence is not None
    assert persisted.explanation_text is not None


@pytest.mark.asyncio
async def test_false_claim_generates_evidence_and_contradictions(db_session):
    job = Job(
        job_id="job-false-claim",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="Water boils at 20 degree celcius",
    )
    db_session.add(job)
    db_session.commit()

    pipeline = PipelineService(db_session, dev_mode=False)
    summary = await pipeline.execute(job.job_id)

    assert summary["completed"] is True
    assert summary["verified_answer"] == "Water boils at 100°C at standard atmospheric pressure."
    assert any("100°C" in e for e in summary["retrieved_evidence"]["evidence"])
    assert summary["retrieved_evidence"]["supporting_documents"][0] == "Wikipedia - Boiling point"
    assert len(summary["retrieved_evidence"]["contradictions"]) >= 1
    assert "20" in summary["retrieved_evidence"]["contradictions"][0]


@pytest.mark.asyncio
async def test_stage_failure_updates_stage_status(db_session):
    job = Job(
        job_id="job-failure-run",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="Trigger failure",
    )
    db_session.add(job)
    db_session.commit()

    pipeline = PipelineService(db_session, dev_mode=False)
    pipeline.max_stage_retries = 1

    async def failing_handler(state):
        raise RuntimeError("stage failure")

    with pytest.raises(RuntimeError, match="stage failure"):
        await pipeline._run_stage(
            job,
            stage_number=1,
            stage_name="Failing Stage",
            progress=10,
            progress_callback=None,
            pipeline_state={"job_id": job.job_id},
            handler=failing_handler,
        )

    failed_stage = db_session.query(Stage).filter(Stage.job_id == job.job_id, Stage.stage_number == 1).one()
    assert failed_stage.status == "failed"
    assert "stage failure" in failed_stage.error_message


@pytest.mark.asyncio
async def test_concurrent_execution_for_two_jobs(engine):
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )

    session = SessionLocal()
    job_one = Job(
        job_id="job-concurrent-1",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="Test concurrent 1",
    )
    job_two = Job(
        job_id="job-concurrent-2",
        status=JobStatus.PENDING.value,
        input_type="text",
        input_text="Test concurrent 2",
    )
    session.add_all([job_one, job_two])
    session.commit()
    session.close()

    async def run_job(job_id):
        local_session = SessionLocal()
        pipeline = PipelineService(local_session, dev_mode=False)
        result = await pipeline.execute(job_id)
        local_session.close()
        return result

    first, second = await asyncio.gather(
        run_job(job_one.job_id),
        run_job(job_two.job_id),
    )

    assert first["completed"] is True
    assert second["completed"] is True

    verifier = SessionLocal()
    persisted = verifier.query(Result).filter(Result.job_id.in_([job_one.job_id, job_two.job_id])).all()
    assert len(persisted) == 2
    verifier.close()
