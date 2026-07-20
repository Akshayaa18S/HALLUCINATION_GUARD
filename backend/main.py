"""
Main FastAPI application entry point
"""
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from middleware.rate_limiter import RateLimiterMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from config import settings
from database import init_db, get_db
from schemas.job_schemas import AnalyzeRequest, JobResponse
from services.job_manager import JobManager
from services.websocket_manager import progress_websocket_manager
from services.queue_manager import job_queue_manager
from services.knowledge_base import knowledge_base


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages app startup and shutdown
    """
    # Startup
    print("🚀 Starting HALLUCINATION_GUARD Backend...")
    init_db()
    # Build/load the FAISS knowledge base once so the first analysis request
    # isn't slowed down by embedding + index construction.
    try:
        knowledge_base.load()
        print("✓ Knowledge base ready")
    except Exception as exc:  # pragma: no cover - best-effort warmup
        print(f"⚠ Knowledge base warmup failed, will retry lazily on first request: {exc}")
    await job_queue_manager.start()
    print("✓ Application started successfully")
    
    yield
    
    # Shutdown
    await job_queue_manager.stop()
    print("🛑 Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time Hallucination Detection Pipeline",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add simple in-memory rate limiter (dev only)
app.add_middleware(RateLimiterMiddleware, max_requests_per_minute=settings.RATE_LIMIT_RPM)


# ==================== API ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "running",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "dev_mode": settings.DEV_MODE
    }


@app.post("/api/analyze", response_model=JobResponse, tags=["Analysis"])
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, db = Depends(get_db)):
    """
    Submit text/image for hallucination analysis
    
    Returns:
        - job_id: Unique identifier for tracking progress
        - status: Initial status (pending)
        - created_at: When the job was created
        
    Frontend should subscribe to WebSocket: `/ws/progress/{job_id}`
    """
    try:
        # Validate input
        if not request.input_text and not request.input_image_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either input_text or input_image_path must be provided"
            )
        
        # Create job
        job = JobManager.create_job(db, request)

        # Enqueue the job so the worker can execute the pipeline asynchronously.
        await job_queue_manager.enqueue(job.job_id)
        
        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create analysis job: {str(e)}"
        )


@app.get("/api/job/{job_id}", tags=["Job Management"])
async def get_job_status(job_id: str, db = Depends(get_db)):
    """
    Get current status of a job
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Job status, progress, and metadata
    """
    try:
        job = JobManager.get_job(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobManager.get_job_status(db, job_id)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


@app.get("/api/result/{job_id}", tags=["Results"])
async def get_result(job_id: str, db = Depends(get_db)):
    """
    Get final analysis result
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Complete analysis result with explanations and evidence
    """
    try:
        # Get result from database
        from models.result import Result
        result = db.query(Result).filter(Result.job_id == job_id).first()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Result for job {job_id} not found"
            )
        
        return {
            "job_id": result.job_id,
            "user_query": result.user_query,
            "generated_response": result.generated_response,
            "hallucination_probability": result.hallucination_score / 100 if result.hallucination_score is not None else None,
            "confidence": result.confidence,
            "is_hallucination": result.is_hallucination,
            "verified_answer": result.verified_answer,
            "retrieved_evidence": result.retrieved_evidence,
            "explanation": result.explanation_text,
            "execution_pipeline": result.execution_pipeline,
            "total_processing_time_ms": result.total_processing_time_ms,
            "created_at": result.created_at,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve result: {str(e)}"
        )


@app.delete("/api/job/{job_id}", tags=["Job Management"])
async def cancel_job(job_id: str, db = Depends(get_db)):
    """
    Cancel a running job
    
    Args:
        job_id: Unique job identifier
        
    Returns:
        Cancellation status
    """
    try:
        job = JobManager.get_job(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        cancelled_job = JobManager.cancel_job(db, job_id)
        
        return {
            "job_id": cancelled_job.job_id,
            "status": cancelled_job.status,
            "cancelled_at": cancelled_job.completed_at
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


# ==================== WebSocket Endpoints ====================
@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Stream live job progress updates to the frontend."""
    await progress_websocket_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_websocket_manager.disconnect(job_id, websocket)
    except Exception:
        progress_websocket_manager.disconnect(job_id, websocket)


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler
    """
    return JSONResponse(
        content={"error": exc.detail, "status_code": exc.status_code},
        status_code=exc.status_code,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
