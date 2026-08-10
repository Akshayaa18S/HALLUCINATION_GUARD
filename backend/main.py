"""
Phase 1 entrypoint.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or:
    python main.py
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import analyze, auth, health, jobs
from config.settings import settings
from core.exceptions import AppError
from core.logging import configure_logging
from database.base import init_db

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (%s)", settings.app_name, settings.app_env)
    await init_db()
    logger.info("Database ready")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    """Phase 12: lightweight per-request performance logging."""
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(auth.router)
app.include_router(analyze.router)


@app.post("/predict")
async def root_predict(request: Request):
    data = await request.json()
    prompt = data.get("prompt") or data.get("query")
    if not prompt:
        return JSONResponse(status_code=400, content={"detail": "Missing 'prompt' or 'query' field."})
    from predict import MultiHaluDetPredictor
    from api.routes.analyze import _get_predictor
    predictor = _get_predictor()
    return predictor.predict(prompt)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)