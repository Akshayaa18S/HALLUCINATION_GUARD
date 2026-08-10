"""
Database engine + session setup (async SQLAlchemy 2.0 style).

Phase 1 only needs this plus the Job model. Phase 2 adds the Result and
Pipeline tables on top of the same Base/engine defined here.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Create tables that don't exist yet. Called once at app startup."""
    def _migrate(sync_conn):
        from sqlalchemy import inspect, text
        Base.metadata.create_all(sync_conn)
        inspector = inspect(sync_conn)
        if "users" in inspector.get_table_names():
            cols = [c["name"] for c in inspector.get_columns("users")]
            if "hashed_password" not in cols:
                if "password_hash" in cols:
                    sync_conn.execute(text("ALTER TABLE users RENAME COLUMN password_hash TO hashed_password"))
                else:
                    sync_conn.execute(text("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255)"))

    async with engine.begin() as conn:
        await conn.run_sync(_migrate)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a session, guarantees close/rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
