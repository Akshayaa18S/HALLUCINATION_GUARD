import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.models.base import Base
from backend.models.job import Job
from backend.models.stage import Stage
from backend.models.result import Result
from backend.services import pipeline_service as pipeline_service_module
from backend.tests.llm_fakes import fake_generate_response, fake_verify_and_detect_hallucination


@pytest.fixture(autouse=True)
def mock_llm_service(monkeypatch):
    """Replace live Anthropic calls with deterministic fakes for the whole test suite.

    Patches the llm_service module object that pipeline_service.py itself holds a
    reference to (via `from services import llm_service`), rather than a freshly
    imported `backend.services.llm_service`, since bare (`services.llm_service`) and
    package-qualified (`backend.services.llm_service`) imports can resolve to two
    distinct module objects in sys.modules depending on how pytest was invoked.
    """
    monkeypatch.setattr(pipeline_service_module.llm_service, "generate_response", fake_generate_response)
    monkeypatch.setattr(
        pipeline_service_module.llm_service, "verify_and_detect_hallucination", fake_verify_and_detect_hallucination
    )


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test_hallucination_guard.db"
    sqlite_url = f"sqlite:///{db_path}"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(engine):
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
