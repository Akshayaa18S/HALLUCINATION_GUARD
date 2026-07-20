"""
Job model for tracking analysis requests
"""
from sqlalchemy import Column, String, DateTime, Integer, Text
from datetime import datetime
from enum import Enum as PyEnum
import uuid

from .base import Base


class JobStatus(PyEnum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InputType(PyEnum):
    """Input data type"""
    TEXT = "text"
    IMAGE = "image"
    TEXT_IMAGE = "text_image"


class Job(Base):
    """
    Job model for tracking analysis requests
    
    Attributes:
        job_id: Unique identifier for the job
        status: Current job status (pending, running, completed, failed, cancelled)
        input_type: Type of input (text, image, or text_image)
        user_id: Optional user identifier
        input_text: Input text for analysis
        input_image_path: Path to input image file
        created_at: When the job was created
        started_at: When the job execution started
        completed_at: When the job completed
        error_message: Error message if job failed
        retry_count: Number of retry attempts
    """
    
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default=JobStatus.PENDING.value, nullable=False)
    input_type = Column(String, default=InputType.TEXT.value, nullable=False)
    user_id = Column(String, nullable=True)
    
    # Input data
    input_text = Column(Text, nullable=True)
    input_image_path = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    def __repr__(self):
        return f"<Job {self.job_id} - {self.status}>"
