import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datetime import datetime

from backend.schemas.event_schemas import StageEvent, WebSocketMessage, FinalResultEvent


def test_stage_event_validation():
    payload = {
        "job_id": "job123",
        "stage": 2,
        "name": "Generating Response",
        "status": "running",
        "progress_percentage": 20.0,
        "start_time": datetime.utcnow(),
        "end_time": None,
        "duration_ms": None,
        "metadata": {"model": "Llama-3"},
        "error_message": None,
    }

    event = StageEvent(**payload)
    assert event.job_id == "job123"
    assert event.status == "running"
    assert event.metadata["model"] == "Llama-3"


def test_websocket_message_wrapper():
    data = {"job_id": "job123", "stage": 1, "status": "running"}
    message = WebSocketMessage(
        message_type="stage_progress",
        data=data,
        timestamp=datetime.utcnow(),
    )

    assert message.message_type == "stage_progress"
    assert message.data["job_id"] == "job123"


def test_final_result_event_validation():
    final = FinalResultEvent(
        job_id="job123",
        status="completed",
        hallucination=True,
        confidence=0.95,
        generated_response="Generated answer",
        verified_answer="Verified answer",
        retrieved_evidence={"source": "Wikipedia"},
        explanation="explanation text",
        processing_time_ms=4200.0,
    )

    assert final.status == "completed"
    assert final.confidence == 0.95
    assert final.retrieved_evidence["source"] == "Wikipedia"


def test_sample_payloads_are_valid_json():
    import json

    sample_path = os.path.join(os.path.dirname(__file__), "sample_event_payloads.json")
    with open(sample_path, "r", encoding="utf-8") as fh:
        payloads = json.load(fh)

    stage_message = WebSocketMessage(**payloads["stage_progress_example"])
    result_message = WebSocketMessage(**payloads["final_result_example"])

    assert stage_message.message_type == "stage_progress"
    assert result_message.message_type == "result"
