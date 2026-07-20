import sys
import os
import asyncio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
BACKEND_DIR = os.path.abspath(os.path.join(ROOT, 'backend'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient

# Ensure enqueue is a no-op during tests
import backend.services.queue_manager as qmgr

async def _noop_enqueue(job_id: str):
    return None

qmgr.job_queue_manager.enqueue = _noop_enqueue

from backend.database import init_db
from backend.main import app


def test_post_analyze_creates_job():
    init_db()
    with TestClient(app) as client:
        payload = {"input_text": "Test analysis text"}
        resp = client.post("/api/analyze", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in {"pending", "running"}


def test_get_result_not_found():
    init_db()
    with TestClient(app) as client:
        resp = client.get("/api/result/nonexistent-job-id-xyz")

    assert resp.status_code == 404
