import pytest
from evaluation.metrics import compute_classification_metrics
from evaluation.statistics import evaluate_significance, mcnemar_test
from evaluation.calibration import compute_calibration_metrics
from evaluation.performance import PerformanceProfiler
from evaluation.report_generator import EvaluationReportGenerator
from experiment_logging.experiment_logger import ExperimentLogger


def test_metrics_computation():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 1, 0, 1, 1]
    y_prob = [0.9, 0.85, 0.1, 0.6, 0.95]

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 0.80
    assert m["precision"] == 0.75
    assert m["recall"] == 1.00
    assert m["f1"] > 0.80


def test_statistical_significance():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    y_base = [1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
    y_model = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]

    sig = evaluate_significance(y_true, y_base, y_model)
    assert sig.comparison_accuracy > sig.baseline_accuracy
    assert 0.0 <= sig.mcnemar_p_value <= 1.0


def test_calibration_computation():
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.2, 0.1]
    calib = compute_calibration_metrics(y_true, y_prob)

    assert calib.ece >= 0.0
    assert calib.brier_score < 0.10
    assert len(calib.reliability_bins) == 10


def test_performance_profiler():
    profiler = PerformanceProfiler()
    summary = profiler.get_summary(retrieval_ms=10.0, verification_ms=15.0, total_ms=50.0)

    assert summary.retrieval_ms == 10.0
    assert summary.verification_ms == 15.0
    assert summary.total_ms == 50.0
    assert summary.memory_mb > 0.0


from evaluation.benchmark import run_benchmark_suite


def test_benchmark_evaluator():
    res = run_benchmark_suite()
    assert "overall_metrics" in res
    assert "calibration_metrics" in res
    assert "retrieval_metrics" in res
    assert res["overall_metrics"]["accuracy"] >= 0.0


def test_experiment_logger_and_reports(tmp_path):
    logger = ExperimentLogger(reports_dir=tmp_path / "experiments")
    metrics = {"accuracy": 0.95, "f1": 0.94}
    cfg = {"claim_verification": True}
    perf = {"total_ms": 42.0, "memory_mb": 250.0}

    log_path = logger.log_experiment("TestSet", metrics, cfg, perf)
    assert log_path.exists()

    gen = EvaluationReportGenerator(reports_dir=tmp_path)
    reports = gen.generate_all_reports("TestSet", metrics, performance_summary=perf)
    assert reports["json"].exists()
    assert reports["csv"].exists()
    assert reports["markdown"].exists()
