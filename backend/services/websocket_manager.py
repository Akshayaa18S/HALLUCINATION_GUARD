"""
In-memory WebSocket connection manager for streaming pipeline progress.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Set

from fastapi import WebSocket


class ProgressWebSocketManager:
    """Tracks active WebSocket connections by job_id and broadcasts updates."""

    def __init__(self) -> None:
        self._connections: dict[str, Set[WebSocket]] = defaultdict(set)
        self._latest_events: dict[str, Dict[str, Any]] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id].add(websocket)
        if job_id in self._latest_events:
            await websocket.send_json(self._latest_events[job_id])

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        connections = self._connections.get(job_id)
        if not connections:
            return
        connections.discard(websocket)
        if not connections:
            self._connections.pop(job_id, None)

    async def broadcast(self, job_id: str, payload: Dict[str, Any]) -> None:
        self._latest_events[job_id] = payload
        connections = list(self._connections.get(job_id, set()))
        if not connections:
            return

        stale_connections: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale_connections.append(websocket)

        for websocket in stale_connections:
            self.disconnect(job_id, websocket)

    def get_latest(self, job_id: str) -> Dict[str, Any] | None:
        return self._latest_events.get(job_id)


progress_websocket_manager = ProgressWebSocketManager()
