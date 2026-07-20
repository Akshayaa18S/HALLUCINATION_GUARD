"""
Logging configuration and utilities (PHASE 10: DEBUG_LOGGING_ENABLED support)
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging(name: str = "hallucination_guard", debug_enabled: bool = False):
    """
    Set up logging configuration
    
    Args:
        name: Logger name
        debug_enabled: Enable debug logging level
        
    Returns:
        Configured logger instance
    """
    from config import settings

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler level depends on debug mode
    console_level = logging.DEBUG if (debug_enabled or settings.DEBUG_LOGGING_ENABLED) else logging.INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # File handler
    log_file = LOGS_DIR / f"hallucination_guard_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    if debug_enabled or settings.DEBUG_LOGGING_ENABLED:
        logger.debug("Debug logging enabled")
    
    return logger


# Global logger instance
logger = setup_logging()


def log_stage_start(job_id: str, stage_number: int, stage_name: str):
    """Log stage start"""
    logger.info(f"[Job {job_id}] Stage {stage_number} ({stage_name}) STARTED")


def log_stage_complete(job_id: str, stage_number: int, stage_name: str, duration_ms: float):
    """Log stage completion"""
    logger.info(f"[Job {job_id}] Stage {stage_number} ({stage_name}) COMPLETED in {duration_ms}ms")


def log_stage_error(job_id: str, stage_number: int, stage_name: str, error: str):
    """Log stage error"""
    logger.error(f"[Job {job_id}] Stage {stage_number} ({stage_name}) FAILED: {error}")


def log_job_complete(job_id: str, total_duration_ms: float):
    """Log job completion"""
    logger.info(f"[Job {job_id}] ANALYSIS COMPLETE in {total_duration_ms}ms")


def log_job_error(job_id: str, error: str):
    """Log job error"""
    logger.error(f"[Job {job_id}] ANALYSIS FAILED: {error}")
