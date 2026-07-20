import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from backend.services.websocket_manager import ProgressWebSocketManager


class DummyWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_progress_websocket_manager_broadcasts_to_connected_client():
    manager = ProgressWebSocketManager()
    websocket = DummyWebSocket()

    await manager.connect("job-ws-1", websocket)
    assert websocket.accepted is True

    payload = {"job_id": "job-ws-1", "stage": 1, "status": "running"}
    await manager.broadcast("job-ws-1", payload)
    assert websocket.sent == [payload]

    manager.disconnect("job-ws-1", websocket)
    await manager.broadcast("job-ws-1", payload)
    assert websocket.sent == [payload]


@pytest.mark.asyncio
async def test_connect_sends_latest_event_if_available():
    manager = ProgressWebSocketManager()
    latest_payload = {"job_id": "job-ws-2", "stage": 2, "status": "running"}
    await manager.broadcast("job-ws-2", latest_payload)

    websocket = DummyWebSocket()
    await manager.connect("job-ws-2", websocket)

    assert websocket.sent == [latest_payload]
    assert manager.get_latest("job-ws-2") == latest_payload
