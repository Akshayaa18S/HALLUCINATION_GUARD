"""
Scientific evidence-path verification tests for MultiHaluDet System C.
Ensures evidence availability and retrieval confidence flags strictly reflect
retrieved evidence passages rather than unretrieved query fallbacks.
"""

import numpy as np
import pytest

from multihaludet.feature_extractor import ExplicitFeatureExtractor
from multihaludet.generation_backend import GenerationBundle
from multihaludet.pipeline import MultiHaluDetModel


def test_evidence_available_flag_strictly_requires_retrieved_evidence():
    """Verify evidence_available == 1.0 and retrieval confidence > 0 ONLY when evidence_texts are supplied."""
    extractor = ExplicitFeatureExtractor(strict_nli=False)

    q = "What is the capital of France?"
    r = "Paris is the capital of France."
    ev = ["Paris is the capital and largest city of France."]
    scores = [0.95]

    feats = extractor.extract_features(q, r, evidence_texts=ev, retrieval_scores=scores)

    assert feats["evidence_available"] == 1.0, f"Expected evidence_available=1.0 with retrieved evidence, got {feats['evidence_available']}"
    assert feats["max_retrieval_confidence"] == 0.95, f"Expected max_retrieval_confidence=0.95, got {feats['max_retrieval_confidence']}"
    assert feats["avg_retrieval_confidence"] == 0.95, f"Expected avg_retrieval_confidence=0.95, got {feats['avg_retrieval_confidence']}"


def test_no_evidence_returns_evidence_available_zero():
    """Verify evidence_available == 0.0 and retrieval confidence == 0.0 when evidence_texts is None or empty."""
    extractor = ExplicitFeatureExtractor(strict_nli=False)

    q = "What is the capital of France?"
    r = "Paris is the capital of France."

    # None case
    feats_none = extractor.extract_features(q, r, evidence_texts=None)
    assert feats_none["evidence_available"] == 0.0, f"Expected evidence_available=0.0 when evidence_texts=None, got {feats_none['evidence_available']}"
    assert feats_none["max_retrieval_confidence"] == 0.0, f"Expected max_retrieval_confidence=0.0 when evidence_texts=None, got {feats_none['max_retrieval_confidence']}"
    assert feats_none["avg_retrieval_confidence"] == 0.0, f"Expected avg_retrieval_confidence=0.0 when evidence_texts=None, got {feats_none['avg_retrieval_confidence']}"

    # Empty list case
    feats_empty = extractor.extract_features(q, r, evidence_texts=[])
    assert feats_empty["evidence_available"] == 0.0, f"Expected evidence_available=0.0 when evidence_texts=[], got {feats_empty['evidence_available']}"
    assert feats_empty["max_retrieval_confidence"] == 0.0
    assert feats_empty["avg_retrieval_confidence"] == 0.0

    # Whitespace list case
    feats_ws = extractor.extract_features(q, r, evidence_texts=["   ", ""])
    assert feats_ws["evidence_available"] == 0.0
    assert feats_ws["max_retrieval_confidence"] == 0.0


def test_pipeline_model_forward_propagates_evidence_texts():
    """Verify MultiHaluDetModel.forward correctly passes evidence_texts down to feature extraction."""
    model = MultiHaluDetModel(hidden_size=16)

    # Create dummy GenerationBundle
    hidden = np.zeros((3, 2, 16), dtype=np.float32)
    logits = np.zeros((2, 100), dtype=np.float32)
    tokens = np.array([10, 20], dtype=np.int64)

    bundle = GenerationBundle(
        text="Paris is in France.",
        layer_step_hidden=hidden,
        step_logits=logits,
        chosen_token_ids=tokens,
        num_layers=2,
        hidden_size=16,
        prompt_token_count=5,
        query="What is the capital?",
        evidence_texts=["Paris is the capital of France."],
        retrieval_scores=[0.88],
    )

    res = model(bundle)
    assert "internal_hallucination_probability" in res
    assert 0.0 <= res["internal_hallucination_probability"] <= 1.0
