"""
Async in-memory job queue for orchestrating pipeline execution.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from database import SessionLocal
from services.pipeline_service import PipelineService
from services.websocket_manager import progress_websocket_manager


@dataclass
class QueueJob:
    """Represents a queued analysis job."""

    job_id: str


class JobQueueManager:
    """Simple async queue that runs one worker for demo/dev mode."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueueJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker(), name="hallucination-guard-job-worker")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                if self._worker_task.get_loop() is asyncio.get_running_loop():
                    await self._worker_task
            except asyncio.CancelledError:
                pass
            except RuntimeError:
                # If shutdown is running on a different event loop, do not await across loops
                pass
        self._worker_task = None

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(QueueJob(job_id=job_id))

    async def _worker(self) -> None:
        while self._running:
            queue_job = await self._queue.get()
            db_session = SessionLocal()
            try:
                pipeline = PipelineService(db_session)
                await pipeline.execute(
                    queue_job.job_id,
                    progress_callback=lambda payload, jid=queue_job.job_id: asyncio.create_task(
                        progress_websocket_manager.broadcast(jid, payload)
                    ),
                )
            except Exception as exc:
                await progress_websocket_manager.broadcast(
                    queue_job.job_id,
                    {
                        "type": "error",
                        "data": {
                            "job_id": queue_job.job_id,
                            "error_message": str(exc),
                        },
                    },
                )
            finally:
                db_session.close()
                self._queue.task_done()


job_queue_manager = JobQueueManager()