"""
Regression Test for NumericChecker Ablation Behavior.

Verifies that:
1. NumericChecker alters the feature vector (numeric_relative_error) when numeric entities differ between response and evidence.
2. For non-numeric text, numeric_relative_error defaults to 0.0, explaining why AUROC remains unchanged on non-numeric benchmark subsets.
"""

import pytest
import numpy as np
from multihaludet.feature_extractor import ExplicitFeatureExtractor


def test_numeric_checker_alters_features_on_numeric_mismatch():
    extractor = ExplicitFeatureExtractor()
    query = "What is the height of Mount Everest?"
    resp_correct = "Mount Everest has an elevation of 8848 meters."
    resp_incorrect = "Mount Everest has an elevation of 5200 meters."
    evidence = ["Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation is 8848.86 m."]

    feats_correct = extractor.extract_features(query, resp_correct, evidence_texts=evidence)
    feats_incorrect = extractor.extract_features(query, resp_incorrect, evidence_texts=evidence)

    # Correct number match should have ~0 relative error
    assert feats_correct["numeric_relative_error"] < 0.05

    # Incorrect number mismatch should have significant relative error (>0.3)
    assert feats_incorrect["numeric_relative_error"] > 0.30


def test_numeric_checker_neutral_on_non_numeric_text():
    extractor = ExplicitFeatureExtractor()
    query = "Who wrote Hamlet?"
    resp = "Hamlet was written by William Shakespeare."
    evidence = ["William Shakespeare was an English playwright, poet and actor."]

    feats = extractor.extract_features(query, resp, evidence_texts=evidence)
    assert feats["numeric_relative_error"] == 0.0
