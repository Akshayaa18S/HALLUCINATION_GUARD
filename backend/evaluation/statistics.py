"""Statistical significance testing module for model evaluation comparisons.

Provides McNemar's test, bootstrap confidence interval estimation,
and hypothesis testing for paired classifier evaluations.
"""

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SignificanceTestResult:
    baseline_accuracy: float
    comparison_accuracy: float
    accuracy_improvement: float
    mcnemar_p_value: float
    significant: float
    confidence_interval_95: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "comparison_accuracy": round(self.comparison_accuracy, 4),
            "accuracy_improvement": round(self.accuracy_improvement, 4),
            "mcnemar_p_value": round(self.mcnemar_p_value, 5),
            "significant": bool(self.significant),
            "confidence_interval_95": (
                round(self.confidence_interval_95[0], 4),
                round(self.confidence_interval_95[1], 4),
            ),
        }


def mcnemar_test(y_true: list[int], y_pred_baseline: list[int], y_pred_model: list[int]) -> float:
    """Compute McNemar's test p-value for paired binary predictions."""
    y_t = np.array(y_true, dtype=int)
    b_correct = (np.array(y_pred_baseline, dtype=int) == y_t)
    m_correct = (np.array(y_pred_model, dtype=int) == y_t)

    # b: baseline correct, model wrong (n10)
    # c: baseline wrong, model correct (n01)
    b = int(np.sum(b_correct & ~m_correct))
    c = int(np.sum(~b_correct & m_correct))

    if b + c == 0:
        return 1.0

    # Continuity corrected chi-square statistic
    statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
    # Approximating p-value from chi-square distribution with 1 degree of freedom
    # sf = 1 - cdf for chi2(df=1)
    p_value = math.erfc(math.sqrt(statistic) / math.sqrt(2.0))
    return float(np.clip(p_value, 0.0, 1.0))


def bootstrap_confidence_interval(
    y_true: list[int],
    y_pred: list[int],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for accuracy metric."""
    rng = np.random.default_rng(seed)
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    n = len(y_t)

    if n == 0:
        return (0.0, 0.0)

    accs = []
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        accs.append(float(np.mean(y_t[idxs] == y_p[idxs])))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(accs, alpha * 100))
    upper = float(np.percentile(accs, (1.0 - alpha) * 100))
    return (lower, upper)


def evaluate_significance(
    y_true: list[int],
    y_pred_baseline: list[int],
    y_pred_model: list[int],
    alpha: float = 0.05,
) -> SignificanceTestResult:
    """Perform statistical significance evaluation between baseline and proposed model."""
    y_t = np.array(y_true, dtype=int)
    b_acc = float(np.mean(np.array(y_pred_baseline, dtype=int) == y_t)) if len(y_t) > 0 else 0.0
    m_acc = float(np.mean(np.array(y_pred_model, dtype=int) == y_t)) if len(y_t) > 0 else 0.0

    p_val = mcnemar_test(y_true, y_pred_baseline, y_pred_model)
    ci = bootstrap_confidence_interval(y_true, y_pred_model)

    return SignificanceTestResult(
        baseline_accuracy=b_acc,
        comparison_accuracy=m_acc,
        accuracy_improvement=m_acc - b_acc,
        mcnemar_p_value=p_val,
        significant=p_val < alpha,
        confidence_interval_95=ci,
    )
