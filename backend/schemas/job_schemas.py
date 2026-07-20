"""
Pydantic schemas for job-related requests and responses
"""
from pydantic import BaseModel, Field, root_validator
from typing import Optional, Literal
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """
    Request schema for analyze endpoint
    
    Attributes:
        input_text: Text to analyze
        input_image_path: Path to image file (optional)
        user_id: Optional user identifier
    """
    input_text: Optional[str] = None
    input_image_path: Optional[str] = None
    user_id: Optional[str] = None
    
    class Config:
        description = "Analysis request with text and/or image"

    @root_validator(skip_on_failure=True)
    def at_least_one_input(cls, values):
        text = values.get("input_text")
        img = values.get("input_image_path")
        if not text and not img:
            raise ValueError("Either input_text or input_image_path must be provided")
        return values


class JobResponse(BaseModel):
    """
    Response when job is created
    
    Attributes:
        job_id: Unique job identifier
        status: Current job status
        created_at: When job was created
    """
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    created_at: datetime
    
    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """
    Response for job status query
    
    Attributes:
        job_id: Unique job identifier
        status: Current job status
        input_type: Type of input provided
        progress_percentage: Overall progress 0-100
        current_stage: Current stage number (1-8)
        started_at: When job started
        created_at: When job was created
        retry_count: Number of retries
    """
    job_id: str
    status: str
    input_type: str
    progress_percentage: Optional[float] = None
    current_stage: Optional[int] = None
    current_stage_name: Optional[str] = None
    started_at: Optional[datetime] = None
    created_at: datetime
    retry_count: int
    
    class Config:
        from_attributes = True


class JobCancelResponse(BaseModel):
    """
    Response for job cancellation
    
    Attributes:
        job_id: Job that was cancelled
        status: New status (should be "cancelled")
        cancelled_at: When the job was cancelled
    """
    job_id: str
    status: str
    cancelled_at: datetime
    
    class Config:
        from_attributes = True


class JobErrorResponse(BaseModel):
    """
    Response for job errors
    
    Attributes:
        job_id: Job that failed
        status: Status (should be "failed")
        error_message: Error description
        retry_count: Number of times retried
    """
    job_id: str
    status: str
    error_message: str
    retry_count: int
    
    class Config:
        from_attributes = True
