import pytest
import numpy as np
from multihaludet.feature_extractor import ExplicitFeatureExtractor, get_nli_pipeline


def test_deberta_nli_semantic_behavior():
    """Test 3: Verifies DeBERTa NLI semantic correctness across entailed, contradicted, and neutral pairs."""
    try:
        pipeline = get_nli_pipeline(strict_nli=True)
    except Exception as exc:
        pytest.skip(f"DeBERTa NLI model not available locally ({exc}) - skipping live model test.")

    extractor = ExplicitFeatureExtractor(strict_nli=True)

    # 1. Clearly Entailed Pair
    q1 = "What is the capital of France?"
    r1 = "Paris is the capital of France."
    ev1 = ["Paris is the capital and most populous city of France."]
    f1 = extractor.extract_features(q1, r1, evidence_texts=ev1)

    assert f1["nli_entailment_score"] > 0.60, f"Expected high entailment, got {f1['nli_entailment_score']}"
    assert f1["nli_contradiction_score"] < 0.30, f"Expected low contradiction, got {f1['nli_contradiction_score']}"

    # 2. Clearly Contradicted Pair
    q2 = "What shape is the Earth?"
    r2 = "The Earth is completely flat."
    ev2 = ["The Earth is an oblate spheroid and spherical in shape."]
    f2 = extractor.extract_features(q2, r2, evidence_texts=ev2)

    assert f2["nli_contradiction_score"] > 0.60, f"Expected high contradiction, got {f2['nli_contradiction_score']}"
    assert f2["nli_entailment_score"] < 0.30, f"Expected low entailment, got {f2['nli_entailment_score']}"

    # 3. Unrelated / Neutral Pair
    q3 = "How does quantum computing work?"
    r3 = "Quantum computers use qubits."
    ev3 = ["Baking bread requires flour, water, salt, and yeast."]
    f3 = extractor.extract_features(q3, r3, evidence_texts=ev3)

    assert f3["nli_neutral_score"] > 0.40, f"Expected high neutral score for unrelated pair, got {f3['nli_neutral_score']}"


def test_strict_nli_fail_closed_on_inference_error(monkeypatch):
    """Test 1: Asserts strict_nli=True fails closed (raises RuntimeError) on inference failure."""
    extractor = ExplicitFeatureExtractor(strict_nli=False)
    extractor.strict_nli = True

    def mock_broken_pipeline(*a, **kw):
        raise RuntimeError("Mock GPU CUDA OOM")

    extractor.nli_pipeline = mock_broken_pipeline

    with pytest.raises(RuntimeError, match="STRICT NLI FAIL-CLOSED ERROR"):
        extractor.extract_features("query", "response", evidence_texts=["some evidence"])
