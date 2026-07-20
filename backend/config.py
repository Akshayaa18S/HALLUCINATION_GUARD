"""
Configuration management for HALLUCINATION_GUARD backend
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    
    # FastAPI
    APP_NAME = "HALLUCINATION_GUARD"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hallucination_guard.db")
    SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "False").lower() == "true"
    
    # Redis & Job Queue
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    
    # Job Configuration
    JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
    JOB_MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))
    JOB_CLEANUP_DAYS = int(os.getenv("JOB_CLEANUP_DAYS", "7"))
    
    # Development Mode
    DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"
    STAGE_DELAY_MIN_MS = int(os.getenv("STAGE_DELAY_MIN_MS", "500"))
    STAGE_DELAY_MAX_MS = int(os.getenv("STAGE_DELAY_MAX_MS", "1500"))
    
    # Development Mode Features (PHASE 10)
    DELAY_SIMULATION_ENABLED = os.getenv("DELAY_SIMULATION_ENABLED", "True").lower() == "true"
    DEBUG_LOGGING_ENABLED = os.getenv("DEBUG_LOGGING_ENABLED", "False").lower() == "true"
    MOCK_DATA_MODE = os.getenv("MOCK_DATA_MODE", "False").lower() == "true"
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    
    # LLM Models
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen-VL-Chat")

    # Ollama local backend (preferred for development and demos)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Live LLM backend (used by services/llm_service.py to replace the old
    # hardcoded stage-2/5/6/7 logic with real model output)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    
    # RAG Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    KNOWLEDGE_SOURCES = ["wikipedia", "fever", "halueval"]
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    WIKIPEDIA_RESULTS_PER_QUERY = int(os.getenv("WIKIPEDIA_RESULTS_PER_QUERY", "3"))
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
    # Rate limiting
    RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))


# Global settings instance
settings = Config()
