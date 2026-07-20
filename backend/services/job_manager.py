"""
Job management service for CRUD operations and job lifecycle
"""
from sqlalchemy.orm import Session
from models.job import Job, JobStatus
from models.result import Result
from schemas.job_schemas import AnalyzeRequest, JobResponse, JobStatusResponse
from datetime import datetime
import uuid


class JobManager:
    """
    Service for managing job lifecycle
    """
    
    @staticmethod
    def create_job(db: Session, request: AnalyzeRequest) -> Job:
        """
        Create a new analysis job
        
        Args:
            db: Database session
            request: Analysis request data
            
        Returns:
            Created Job object
        """
        job_id = str(uuid.uuid4())
        
        # Determine input type
        input_type = "text"
        if request.input_text and request.input_image_path:
            input_type = "text_image"
        elif request.input_image_path:
            input_type = "image"
        
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING.value,
            input_type=input_type,
            user_id=request.user_id,
            input_text=request.input_text,
            input_image_path=request.input_image_path
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_job(db: Session, job_id: str) -> Job:
        """
        Get job by ID
        
        Args:
            db: Database session
            job_id: Job identifier
            
        Returns:
            Job object or None
        """
        return db.query(Job).filter(Job.job_id == job_id).first()
    
    @staticmethod
    def get_job_status(db: Session, job_id: str) -> JobStatusResponse:
        """
        Get detailed job status
        
        Args:
            db: Database session
            job_id: Job identifier
            
        Returns:
            JobStatusResponse object
        """
        job = JobManager.get_job(db, job_id)
        
        if not job:
            return None
        
        from models.stage import Stage

        stage = (
            db.query(Stage)
            .filter(Stage.job_id == job_id)
            .order_by(Stage.stage_number.desc())
            .first()
        )
        current_stage = stage.stage_number if stage else None
        current_stage_name = stage.name if stage else None
        progress_percentage = stage.progress_percentage if stage else None
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            input_type=job.input_type,
            progress_percentage=progress_percentage,
            current_stage=current_stage,
            current_stage_name=current_stage_name,
            started_at=job.started_at,
            created_at=job.created_at,
            retry_count=job.retry_count
        )
    
    @staticmethod
    def update_job_status(db: Session, job_id: str, status: str, error_message: str = None) -> Job:
        """
        Update job status
        
        Args:
            db: Database session
            job_id: Job identifier
            status: New status
            error_message: Optional error message
            
        Returns:
            Updated Job object
        """
        job = JobManager.get_job(db, job_id)
        
        if not job:
            return None
        
        job.status = status
        if error_message:
            job.error_message = error_message
        
        if status == JobStatus.RUNNING.value and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            job.completed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def cancel_job(db: Session, job_id: str) -> Job:
        """
        Cancel a job
        
        Args:
            db: Database session
            job_id: Job identifier
            
        Returns:
            Cancelled Job object
        """
        return JobManager.update_job_status(db, job_id, JobStatus.CANCELLED.value)
    
    @staticmethod
    def increment_retry_count(db: Session, job_id: str) -> Job:
        """
        Increment retry count for a job
        
        Args:
            db: Database session
            job_id: Job identifier
            
        Returns:
            Updated Job object
        """
        job = JobManager.get_job(db, job_id)
        
        if job:
            job.retry_count += 1
            db.commit()
            db.refresh(job)
        
        return job
    
    @staticmethod
    def delete_old_jobs(db: Session, days: int = 7) -> int:
        """
        Delete jobs older than specified days
        
        Args:
            db: Database session
            days: Number of days
            
        Returns:
            Number of jobs deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        deleted = db.query(Job).filter(
            Job.created_at < cutoff_date,
            Job.status.in_([
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value
            ])
        ).delete()
        
        db.commit()
        
        return deleted
