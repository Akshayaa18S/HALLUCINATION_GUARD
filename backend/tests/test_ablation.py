import pytest
from evaluation.ablation import AblationFramework


def test_ablation_configs():
    framework = AblationFramework()
    assert len(framework.ablation_configs) == 8
    assert "full_framework" in framework.ablation_configs
    assert "baseline_nli_only" in framework.ablation_configs


def test_ablation_study_runner():
    framework = AblationFramework()
    samples = [
        {"query": "q1", "label": 1, "simulated_prob": 0.90, "nli_prob": 0.85, "retrieval_prob": 0.80},
        {"query": "q2", "label": 0, "simulated_prob": 0.10, "nli_prob": 0.15, "retrieval_prob": 0.12},
    ]

    res = framework.run_ablations(samples)
    assert "full_framework" in res
    assert "baseline_nli_only" in res
    assert "accuracy" in res["full_framework"]

