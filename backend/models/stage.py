"""
Stage model for tracking pipeline execution stages
"""
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, ForeignKey, JSON
from enum import Enum as PyEnum

from .base import Base


class StageStatus(PyEnum):
    """Stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Stage(Base):
    """
    Stage model for tracking pipeline execution stages (1-8)
    
    Attributes:
        id: Auto-incremented primary key
        job_id: Reference to parent job
        stage_number: Stage number (1-8)
        name: Human-readable stage name
        status: Current stage status
        progress_percentage: Progress 0-100
        start_time: When stage started
        end_time: When stage completed
        duration_ms: Duration in milliseconds
        error_message: Error message if failed
        metadata_json: Stage-specific metadata as JSON
    """
    
    __tablename__ = "stages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.job_id"), nullable=False)
    
    # Stage info
    stage_number = Column(Integer, nullable=False)  # 1-8
    name = Column(String, nullable=False)
    status = Column(String, default=StageStatus.PENDING.value, nullable=False)
    
    # Progress tracking
    progress_percentage = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Duration in milliseconds
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Metadata
    metadata_json = Column(JSON, nullable=True)  # Stage-specific metadata
    
    def __repr__(self):
        return f"<Stage {self.stage_number} ({self.name}) - {self.status}>"
