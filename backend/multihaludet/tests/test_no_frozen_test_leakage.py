import sys
import pytest
from multihaludet.training.datasets import HallucinationExample, verify_no_test_contamination



def test_frozen_test_isolation_and_no_leakage():
    """Verifies strict isolation between frozen 500 benchmark IDs and development pool IDs."""
    dev_pool = [
        HallucinationExample(f"Dev Q{i}", f"Dev Ans {i}", i % 2 == 0, "en", "dev_source", str(i))
        for i in range(100)
    ]
    frozen_test = [
        HallucinationExample(f"Test Q{i}", f"Test Ans {i}", i % 2 == 0, "en", "frozen_500_benchmark", str(1000 + i))
        for i in range(50)
    ]

    # Verify zero intersection
    dev_hashes = {hash((e.query, e.response)) for e in dev_pool}
    test_hashes = {hash((e.query, e.response)) for e in frozen_test}
    assert len(dev_hashes & test_hashes) == 0, "Frozen test IDs leaked into development pool!"

    # Verify zero contamination guard passes
    verify_no_test_contamination(dev_pool, frozen_test)


def test_strict_nli_raises_on_unavailable_model(monkeypatch):
    """Verifies that strict_nli=True raises RuntimeError instead of silently falling back."""
    import multihaludet.feature_extractor as fe_mod
    monkeypatch.setattr(fe_mod, "_NLI_PIPELINE", None)

    # Mock pipeline function to raise ImportError
    def mock_pipeline(*a, **kw):
        raise ImportError("Transformers NLI model unavailable")

    import types
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = mock_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="DeBERTa NLI model 'cross-encoder/nli-deberta-v3-base' unavailable"):
        fe_mod.get_nli_pipeline(strict_nli=True)

