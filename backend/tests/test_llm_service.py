import json
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backend.services.llm_service as llm_service


def test_generate_response_prefers_ollama(monkeypatch):
    monkeypatch.setattr(llm_service.settings, "OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(llm_service.settings, "OLLAMA_MODEL", "llama3.2:3b")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"response": "Ollama says hi"}'

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "llama3.2:3b"
        assert payload["stream"] is False
        return FakeResponse()

    monkeypatch.setattr(llm_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_service, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("Anthropic should not be used")))

    response = llm_service.generate_response("Hello from tests")

    assert response == "Ollama says hi"
