import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.services import pipeline_service as pipeline_service_module
from backend.tests.llm_fakes import fake_generate_response, fake_verify_and_detect_hallucination


@pytest.fixture(autouse=True)
def mock_llm_service(monkeypatch):
    """Replace live Anthropic calls with deterministic fakes for the whole test suite.

    See backend/tests/conftest.py for why this patches the llm_service module
    object held by pipeline_service.py directly, rather than a freshly imported
    backend.services.llm_service.
    """
    monkeypatch.setattr(pipeline_service_module.llm_service, "generate_response", fake_generate_response)
    monkeypatch.setattr(
        pipeline_service_module.llm_service, "verify_and_detect_hallucination", fake_verify_and_detect_hallucination
    )
