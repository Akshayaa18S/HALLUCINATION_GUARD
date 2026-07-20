"""
Pydantic schemas for real-time events and WebSocket messages
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal, TypedDict
from datetime import datetime


class StageMetadata(TypedDict, total=False):
    input_received: bool
    input_type: str
    generated_response: str
    hidden_states: Dict[str, Any]
    extracted_features: Dict[str, Any]
    hallucination_result: Dict[str, Any]
    retrieved_evidence: Dict[str, Any]
    explanation: Dict[str, Any]
    analysis_completed: bool


class StageEvent(BaseModel):
    """
    Real-time stage progress event for WebSocket
    
    Attributes:
        job_id: Job identifier
        stage: Stage number (1-8)
        name: Stage name
        status: Stage status (pending, running, completed, failed)
        progress_percentage: Progress 0-100%
        start_time: When stage started
        end_time: When stage completed
        duration_ms: Duration in milliseconds
        metadata: Stage-specific metadata
        error_message: Error message if failed
    """
    job_id: str = Field(..., description="Job identifier")
    stage: int = Field(..., ge=1, le=8, description="Stage number 1-8")
    name: str = Field(..., description="Stage name")
    status: Literal["pending", "running", "completed", "failed"] = Field(..., description="Stage status")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress 0-100%")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123xyz",
                "stage": 2,
                "name": "Generating Response",
                "status": "running",
                "progress_percentage": 20,
                "start_time": "2026-07-15T10:30:00Z",
                "duration_ms": 1500,
                "metadata": {"model": "Llama-3"}
            }
        }


class FinalResultEvent(BaseModel):
    """
    Final analysis result event
    
    Attributes:
        job_id: Job identifier
        status: Completion status (completed or failed)
        hallucination: Whether hallucination was detected
        confidence: Confidence score 0-1
        generated_response: LLM-generated response
        verified_answer: Corrected answer
        retrieved_evidence: Retrieved evidence
        explanation: Explainability text
        processing_time_ms: Total processing time
    """
    job_id: str
    status: Literal["completed", "failed"] = "completed"
    hallucination: bool
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    generated_response: str
    verified_answer: Optional[str] = None
    retrieved_evidence: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123xyz",
                "status": "completed",
                "hallucination": True,
                "confidence": 0.94,
                "generated_response": "Paris is the capital of Germany.",
                "verified_answer": "Paris is the capital of France.",
                "retrieved_evidence": {"source": "Wikipedia", "claim": "Paris is the capital of France"},
                "explanation": "The model incorrectly identified the capital country.",
                "processing_time_ms": 4200.0
            }
        }


class WebSocketMessagePayload(TypedDict):
    message_type: Literal["stage_progress", "result", "error", "heartbeat"]
    data: Dict[str, Any]
    timestamp: str


class WebSocketMessage(BaseModel):
    """
    Generic WebSocket message wrapper
    
    Attributes:
        message_type: Type of message (stage_progress, result, error)
        data: Message payload
        timestamp: When message was sent
    """
    message_type: Literal["stage_progress", "result", "error", "heartbeat"]
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_type": "stage_progress",
                "data": {
                    "job_id": "abc123xyz",
                    "stage": 3,
                    "name": "Hidden State Extraction",
                    "status": "running",
                    "progress_percentage": 35
                },
                "timestamp": "2026-07-15T10:30:02Z"
            }
        }


class ErrorEvent(BaseModel):
    """
    Error event for WebSocket communication
    
    Attributes:
        job_id: Job identifier
        stage: Stage number where error occurred
        error_message: Error description
        timestamp: When error occurred
    """
    job_id: str
    stage: Optional[int] = None
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "abc123xyz",
                "stage": 2,
                "error_message": "Failed to load LLM model",
                "timestamp": "2026-07-15T10:30:05Z"
            }
        }
