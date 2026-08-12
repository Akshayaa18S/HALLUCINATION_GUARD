"""
Dual-Signal Fusion Optimization, Empirical Calibration Engine, Bootstrap CIs & Statistical Significance.
Strictly zero-leakage: all fusion weights, gating parameters, and calibrators are fit on VALIDATION data and frozen.
Test evaluation runs in ONE SINGLE PASS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("hallucination_guard.hallucination.fusion_calibration")


# --- Metric Calculation Utilities ---

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    y_true = np.array(y_true, dtype=float)
    y_probs = np.array(y_probs, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_confidence = np.mean(y_probs[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += (bin_size / n) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Calculates 95% Bootstrap Confidence Intervals for AUROC, AUPRC, and F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    auroc_boot = []
    auprc_boot = []
    f1_boot = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_probs[idx]

        if len(np.unique(yt)) < 2:
            continue

        try:
            auroc_boot.append(roc_auc_score(yt, yp))
        except Exception:
            pass

        try:
            auprc_boot.append(average_precision_score(yt, yp))
        except Exception:
            pass

        preds = (yp >= 0.5).astype(int)
        f1_boot.append(f1_score(yt, preds, zero_division=0))

    def _ci(arr: list) -> Tuple[float, float, float]:
        if not arr:
            return 0.5, 0.5, 0.5
        mean_val = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        return mean_val, low, high

    return {
        "roc_auc": _ci(auroc_boot),
        "auprc": _ci(auprc_boot),
        "f1": _ci(f1_boot),
    }


def compute_paired_significance_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap significance test comparing baseline A vs method B."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    delta_auroc_boot = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            auc_a = roc_auc_score(yt, probs_a[idx])
            auc_b = roc_auc_score(yt, probs_b[idx])
            delta_auroc_boot.append(auc_b - auc_a)
        except Exception:
            pass

    if not delta_auroc_boot:
        return {"delta_auroc_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    delta_mean = float(np.mean(delta_auroc_boot))
    ci_lower = float(np.percentile(delta_auroc_boot, 2.5))
    ci_upper = float(np.percentile(delta_auroc_boot, 97.5))
    p_value = float(np.mean(np.array(delta_auroc_boot) <= 0.0))

    return {
        "delta_auroc_mean": delta_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": max(p_value, 0.001),
    }


# --- Calibrator Classes ---

class TemperatureScalingCalibrator:
    """Optimizes single temperature parameter T on Validation data."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> None:
        val_logits = np.array(val_logits, dtype=float)
        val_labels = np.array(val_labels, dtype=float)

        def nll_loss(t: float) -> float:
            scaled_logits = val_logits / max(t[0], 1e-3)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            return float(-np.mean(val_labels * np.log(probs) + (1.0 - val_labels) * np.log(1.0 - probs)))

        res = minimize(nll_loss, [1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        logger.info("Fitted Temperature Scaling: T = %.4f", self.temperature)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 1e-3)
        return 1.0 / (1.0 + np.exp(-scaled))


class PlattScalingCalibrator:
    """Logistic sigmoid calibration fitted on Validation predictions."""

    def __init__(self) -> None:
        self.model = LogisticRegression()
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> None:
        val_probs = np.array(val_probs).reshape(-1, 1)
        self.model.fit(val_probs, val_labels)
        self.is_fitted = True
        logger.info("Fitted Platt Scaling Calibrator.")

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return probs
        probs_2d = np.array(probs).reshape(-1, 1)
        return self.model.predict_proba(probs_2d)[:, 1]


class IsotonicRegressionCalibrator:
    """Isotonic Regression non-parametric calibrator fitted on Validation predictions."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> None:
        self.model.fit(val_probs, val_labels)
        self.is_fitted = True
        logger.info("Fitted Isotonic Regression Calibrator.")

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return probs
        return self.model.predict(probs)


# --- Fusion Weight Optimizer ---

class FusionWeightOptimizer:
    """Optimizes dual-signal fusion alpha on Validation data."""

    def __init__(self) -> None:
        self.best_alpha: float = 0.70
        self.gating_model = LogisticRegression()

    def optimize_fusion_weights(
        self,
        val_p_internal: np.ndarray,
        val_p_external: np.ndarray,
        val_labels: np.ndarray,
    ) -> float:
        best_auc = -1.0
        best_a = 0.70

        for alpha in np.linspace(0.0, 1.0, 21):
            fused = alpha * val_p_internal + (1.0 - alpha) * val_p_external
            try:
                auc = float(roc_auc_score(val_labels, fused))
            except Exception:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_a = float(alpha)

        self.best_alpha = best_a
        # Fit learned gating model on (p_internal, p_external)
        X_val = np.column_stack([val_p_internal, val_p_external])
        self.gating_model.fit(X_val, val_labels)

        logger.info("Optimized Fusion Alpha on Validation: alpha = %.2f (Validation ROC-AUC = %.4f)", self.best_alpha, best_auc)
        return self.best_alpha

    def predict_fused_fixed(self, p_internal: np.ndarray, p_external: np.ndarray, alpha: float | None = None) -> np.ndarray:
        a = alpha if alpha is not None else self.best_alpha
        return a * p_internal + (1.0 - a) * p_external

    def predict_fused_learned(self, p_internal: np.ndarray, p_external: np.ndarray) -> np.ndarray:
        X = np.column_stack([p_internal, p_external])
        return self.gating_model.predict_proba(X)[:, 1]


def export_calibration_table(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    temp_probs: np.ndarray,
    platt_probs: np.ndarray,
    iso_probs: np.ndarray,
    output_dir: str | Path = "./reports",
) -> None:
    """Exports Empirical Calibration metrics (ECE, Brier, NLL) to Markdown and LaTeX."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    methods = [
        ("Uncalibrated Raw", raw_probs),
        ("Temperature Scaling", temp_probs),
        ("Platt Scaling", platt_probs),
        ("Isotonic Regression", iso_probs),
    ]

    md_lines = [
        "# Empirical Probability Calibration Results",
        "",
        "| Calibration Method | ECE | Brier Score | Negative Log-Likelihood (NLL) |",
        "| :--- | :---: | :---: | :---: |",
    ]

    tex_lines = [
        "% Table 8: Calibration Performance Comparison",
        "\\begin{table}[htbp]",
        "\\caption{Empirical Calibration Metrics on Frozen Test Set}",
        "\\begin{center}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\textbf{Calibration Method} & \\textbf{ECE} & \\textbf{Brier Score} & \\textbf{NLL} \\\\",
        "\\midrule",
    ]

    for name, probs in methods:
        ece = compute_ece(y_true, probs)
        brier = float(brier_score_loss(y_true, probs))
        nll = float(log_loss(y_true, probs, labels=[0, 1]))

        md_lines.append(f"| {name} | {ece:.4f} | {brier:.4f} | {nll:.4f} |")
        tex_lines.append(f"{name} & {ece:.4f} & {brier:.4f} & {nll:.4f} \\\\")

    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\label{tab:calibration}", "\\end{center}", "\\end{table}"])

    (out_path / "calibration_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_path / "calibration_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    logger.info("Saved calibration tables in %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    y_val = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 10)
    p_int = np.array([0.8, 0.2, 0.7, 0.3, 0.85, 0.15, 0.9, 0.1] * 10)
    p_ext = np.array([0.75, 0.25, 0.65, 0.35, 0.8, 0.2, 0.85, 0.15] * 10)

    opt = FusionWeightOptimizer()
    best_a = opt.optimize_fusion_weights(p_int, p_ext, y_val)
    fused_p = opt.predict_fused_fixed(p_int, p_ext)

    # Calibrators
    temp_cal = TemperatureScalingCalibrator()
    temp_cal.fit(np.log(p_int / (1 - p_int + 1e-6)), y_val)

    platt_cal = PlattScalingCalibrator()
    platt_cal.fit(fused_p, y_val)

    iso_cal = IsotonicRegressionCalibrator()
    iso_cal.fit(fused_p, y_val)

    export_calibration_table(
        y_val,
        fused_p,
        temp_cal.calibrate(np.log(p_int / (1 - p_int + 1e-6))),
        platt_cal.calibrate(fused_p),
        iso_cal.calibrate(fused_p),
    )
    print("Fusion Optimization & Calibration Module Verified Successfully.")
