import pytest
from evaluation.ablation import AblationStudyRunner, AblationConfig


def test_ablation_configs():
    configs = AblationStudyRunner.get_standard_ablation_configs()
    assert len(configs) == 6
    assert configs[0].name == "Baseline (MultiHaluDet)"
    assert configs[-1].name == "+ Full Hybrid Framework"


def test_ablation_study_runner():
    runner = AblationStudyRunner()
    samples = [
        {"query": "q1", "label": 1, "prob": 0.90},
        {"query": "q2", "label": 0, "prob": 0.10},
    ]

    res = runner.run_ablation_study(samples, dataset_name="AblationBenchmark")
    assert res["dataset"] == "AblationBenchmark"
    assert "Baseline (MultiHaluDet)" in res["ablation_results"]
    assert "+ Full Hybrid Framework" in res["ablation_results"]
    assert "| **Baseline (MultiHaluDet)** |" in res["comparison_table_markdown"]
