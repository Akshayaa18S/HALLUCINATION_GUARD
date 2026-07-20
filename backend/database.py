"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from config import settings
from typing import Generator
from models.base import Base

# Create database engine
if "sqlite" in settings.DATABASE_URL:
    # For SQLite in development
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.SQLALCHEMY_ECHO
    )
else:
    # For PostgreSQL in production
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.SQLALCHEMY_ECHO,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


def get_db() -> Generator:
    """
    Get database session for dependency injection
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables
    """
    # Import all models to register them
    from models.job import Job
    from models.stage import Stage
    from models.result import Result
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def drop_db():
    """
    Drop all database tables (use with caution)
    """
    Base.metadata.drop_all(bind=engine)
    print("✓ Database tables dropped")
