"""Confidence calibration evaluation module.

Computes Expected Calibration Error (ECE), Brier Score, and reliability bin statistics.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationMetrics:
    ece: float
    brier_score: float
    reliability_bins: list[dict]

    def to_dict(self) -> dict:
        return {
            "expected_calibration_error": round(self.ece, 4),
            "brier_score": round(self.brier_score, 4),
            "reliability_bins": self.reliability_bins,
        }


def compute_calibration_metrics(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Expected Calibration Error (ECE), Brier score, and bin accuracies."""
    if not y_true or not y_prob or len(y_true) != len(y_prob):
        return CalibrationMetrics(ece=0.0, brier_score=0.0, reliability_bins=[])

    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_prob, dtype=float)
    n = len(y_t)

    # Brier score: MSE between predicted probabilities and binary outcomes
    brier = float(np.mean((y_p - y_t) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
        else:
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            avg_acc = float(np.mean(y_t[in_bin]))
            avg_conf = float(np.mean(y_p[in_bin]))
            bin_error = abs(avg_acc - avg_conf)
            ece += (bin_count / n) * bin_error

            bins.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "samples": bin_count,
                "accuracy": round(avg_acc, 4),
                "confidence": round(avg_conf, 4),
                "calibration_error": round(bin_error, 4),
            })
        else:
            bins.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "samples": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_error": 0.0,
            })

    return CalibrationMetrics(ece=float(ece), brier_score=brier, reliability_bins=bins)
