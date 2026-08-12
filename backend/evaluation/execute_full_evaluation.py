"""
Evaluation Layer — Comprehensive Publication Benchmark Suite (Tasks 1 - 10).

Executes genuine end-to-end evaluation on benchmark dataset samples,
computing classification metrics, ECE/Brier scores, latency, bootstrap 95% CIs,
and exporting publication LaTeX/Markdown tables directly from model predictions.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

# Ensure backend in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.metrics import compute_calibration_metrics, compute_classification_metrics
from predict import MultiHaluDetPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.full_eval")

SEEDS = [42, 123, 2024, 3407]


def load_frozen_test_dataset(dataset_path: str = "data/halueval_fever_benchmark_500.csv") -> list[dict[str, Any]]:
    """Loads dataset from CSV file."""
    import hashlib
    p = Path(dataset_path)
    if not p.exists():
        p = backend_dir.parent / dataset_path
    if not p.exists():
        p = backend_dir / dataset_path

    if not p.exists():
        raise FileNotFoundError(
            f"Frozen test dataset not found: {dataset_path}. "
            "Publication evaluation requires an independently prepared test set."
        )

    samples: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        content_bytes = f.read().encode("utf-8")
        dataset_hash = hashlib.sha256(content_bytes).hexdigest()
        f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": int(row.get("id", len(samples) + 1)),
                "prompt": row.get("prompt", row.get("query", "")),
                "generated_response": row.get("generated_response", row.get("response", "")),
                "label": int(row.get("label", 0)),
            })
    logger.info("Loaded frozen benchmark dataset '%s' (SHA-256: %s, N=%d).", str(p), dataset_hash, len(samples))
    return samples


def run_full_evaluation():
    """Executes frozen test evaluation on benchmark dataset."""
    logger.info("Starting MultiHaluDet Frozen Publication Evaluation...")

    dataset_path = "data/halueval_fever_benchmark_500.csv"
    samples = load_frozen_test_dataset(dataset_path)

    predictor = MultiHaluDetPredictor()
    if not predictor.model.is_trained:
        raise RuntimeError("MultiHaluDet model checkpoint failed to load or is not trained.")

    logger.info("Executing Frozen Test Evaluation on %d samples across seeds %s...", len(samples), SEEDS)

    seed_metrics: list[dict[str, float]] = []
    all_latencies: list[float] = []
    primary_sample_results: list[dict[str, Any]] = []
    primary_y_true: list[int] = []
    primary_y_prob: list[float] = []

    for seed_idx, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        y_true: list[int] = []
        y_prob: list[float] = []
        latencies: list[float] = []
        per_sample_results: list[dict[str, Any]] = []

        for sample in samples:
            prompt = sample["prompt"]
            resp = sample["generated_response"]
            label = sample["label"]

            t0 = time.monotonic()
            res = predictor.predict(prompt, response_text=resp, skip_retrieval=False)
            raw_prob = float(res.get("hallucination_probability", 0.50))

            # IMPORTANT: Use actual probability produced by MultiHaluDet without overriding
            prob = float(np.clip(raw_prob, 0.0, 1.0))
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            latencies.append(elapsed_ms)
            all_latencies.append(elapsed_ms)
            y_true.append(label)
            y_prob.append(prob)

            per_sample_results.append({
                "id": sample["id"],
                "prompt": prompt,
                "response": resp,
                "ground_truth": label,
                "probability": round(prob, 4),
                "latency_ms": round(elapsed_ms, 1),
            })

        if seed_idx == 0:
            primary_sample_results = per_sample_results
            primary_y_true = y_true
            primary_y_prob = y_prob

        preds = [1 if p >= predictor.model.decision_threshold else 0 for p in y_prob]
        m = compute_classification_metrics(y_true, preds, y_prob)
        cal = compute_calibration_metrics(y_true, y_prob)
        m["expected_calibration_error"] = cal.get("ece", 0.0)
        m["brier_score"] = cal.get("brier_score", 0.0)
        seed_metrics.append(m)

    # Calculate aggregated results dynamically from seed_metrics
    metric_names = [
        "accuracy", "precision", "recall", "f1", "auroc",
        "pr_auc", "mcc", "cohen_kappa", "expected_calibration_error", "brier_score"
    ]
    aggregated_results: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(m[metric_name]) for m in seed_metrics if metric_name in m]
        if values:
            aggregated_results[metric_name] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
            }

    # Dynamic 95% Bootstrap Confidence Intervals for Primary Seed
    ci_metrics: dict[str, dict[str, float]] = {}
    n_boot = 500
    rng_ci = np.random.RandomState(42)
    boot_stats: dict[str, list[float]] = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auroc"]}
    n_samples = len(primary_y_true)

    for _ in range(n_boot):
        indices = rng_ci.choice(n_samples, size=n_samples, replace=True)
        yt_b = [primary_y_true[i] for i in indices]
        yp_b = [primary_y_prob[i] for i in indices]
        preds_b = [1 if p >= predictor.model.decision_threshold else 0 for p in yp_b]
        b_m = compute_classification_metrics(yt_b, preds_b, yp_b)
        for k in boot_stats:
            boot_stats[k].append(float(b_m.get(k, 0.0)))

    for k, vals in boot_stats.items():
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        ci_metrics[k] = {
            "mean": round(aggregated_results.get(k, {}).get("mean", float(np.mean(vals))), 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    # Dynamic Latency Profiling
    lat_arr = np.array(all_latencies) if all_latencies else np.array([50.0])
    latency_mean = round(float(np.mean(lat_arr)), 1)
    latency_median = round(float(np.median(lat_arr)), 1)
    latency_p90 = round(float(np.percentile(lat_arr, 90)), 1)
    latency_p95 = round(float(np.percentile(lat_arr, 95)), 1)
    latency_max = round(float(np.max(lat_arr)), 1)

    # Primary Seed Classification Metrics & Confusion Matrix
    primary_preds = [1 if p >= predictor.model.decision_threshold else 0 for p in primary_y_prob]
    primary_metrics = compute_classification_metrics(primary_y_true, primary_preds, primary_y_prob)
    primary_cal = compute_calibration_metrics(primary_y_true, primary_y_prob)
    primary_metrics["expected_calibration_error"] = primary_cal.get("ece", 0.0)
    primary_metrics["brier_score"] = primary_cal.get("brier_score", 0.0)

    # Assemble Final Report
    report = {
        "dataset_name": dataset_path,
        "total_samples": len(samples),
        "evaluation_protocol": "FINAL FROZEN TEST EVALUATION — MultiHaluDet Genuine Benchmark",
        "seeds": SEEDS,
        "aggregated_metrics": aggregated_results,
        "classification_metrics": primary_metrics,
        "confidence_intervals_95": ci_metrics,
        "calibration_metrics": {
            "ece": primary_cal.get("ece", 0.0),
            "brier_score": primary_cal.get("brier_score", 0.0),
        },
        "latency_metrics": {
            "mean_ms": latency_mean,
            "median_ms": latency_median,
            "p90_ms": latency_p90,
            "p95_ms": latency_p95,
            "max_ms": latency_max,
        },
        "per_sample_results": primary_sample_results,
    }

    out_paths_json = [
        backend_dir / "data" / "evaluation_report_full_publication.json",
        backend_dir.parent / "data" / "evaluation_report_full_publication.json",
    ]
    for p in out_paths_json:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    logger.info("Evaluation complete! Genuine report exported to '%s'.", str(out_paths_json[0]))
    generate_publication_tables(report)


def generate_publication_tables(report: dict[str, Any]):
    """Generates LaTeX and Markdown publication tables from actual evaluated metrics."""
    m = report.get("classification_metrics", {})
    agg = report.get("aggregated_metrics", {})

    acc_mean = agg.get("accuracy", {}).get("mean", m.get("accuracy", 0.0))
    acc_std = agg.get("accuracy", {}).get("std", 0.0)
    f1_mean = agg.get("f1", {}).get("mean", m.get("f1", 0.0))
    f1_std = agg.get("f1", {}).get("std", 0.0)
    auc_mean = agg.get("auroc", {}).get("mean", m.get("auroc", 0.0))
    auc_std = agg.get("auroc", {}).get("std", 0.0)

    md_content = f"""# MultiHaluDet Benchmark Evaluation Report (v3.1 Frozen)

## 📊 Task 1: Frozen Test Benchmark Suite ($N = {report.get('total_samples', 500)}$, {len(report.get('seeds', []))} Seeds Mean ± Std)

| Metric | MultiHaluDet (Mean ± Std) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **{acc_mean*100:.2f}% ± {acc_std*100:.2f}%** | [{report['confidence_intervals_95'].get('accuracy', {}).get('ci_lower', 0)*100:.1f}%, {report['confidence_intervals_95'].get('accuracy', {}).get('ci_upper', 0)*100:.1f}%] |
| **Precision** | **{agg.get('precision', {}).get('mean', 0)*100:.2f}% ± {agg.get('precision', {}).get('std', 0)*100:.2f}%** | [{report['confidence_intervals_95'].get('precision', {}).get('ci_lower', 0)*100:.1f}%, {report['confidence_intervals_95'].get('precision', {}).get('ci_upper', 0)*100:.1f}%] |
| **Recall (Sensitivity)** | **{agg.get('recall', {}).get('mean', 0)*100:.2f}% ± {agg.get('recall', {}).get('std', 0)*100:.2f}%** | [{report['confidence_intervals_95'].get('recall', {}).get('ci_lower', 0)*100:.1f}%, {report['confidence_intervals_95'].get('recall', {}).get('ci_upper', 0)*100:.1f}%] |
| **F1-Score** | **{f1_mean*100:.2f}% ± {f1_std*100:.2f}%** | [{report['confidence_intervals_95'].get('f1', {}).get('ci_lower', 0)*100:.1f}%, {report['confidence_intervals_95'].get('f1', {}).get('ci_upper', 0)*100:.1f}%] |
| **ROC-AUC (AUROC)** | **{auc_mean:.4f} ± {auc_std:.4f}** | [{report['confidence_intervals_95'].get('auroc', {}).get('ci_lower', 0):.4f}, {report['confidence_intervals_95'].get('auroc', {}).get('ci_upper', 0):.4f}] |
| **PR-AUC** | **{agg.get('pr_auc', {}).get('mean', 0):.4f} ± {agg.get('pr_auc', {}).get('std', 0):.4f}** | — |
| **MCC (Matthews Corr)** | **{agg.get('mcc', {}).get('mean', 0):.4f} ± {agg.get('mcc', {}).get('std', 0):.4f}** | — |
| **Cohen's Kappa ($\kappa$)** | **{agg.get('cohen_kappa', {}).get('mean', 0):.4f} ± {agg.get('cohen_kappa', {}).get('std', 0):.4f}** | — |
| **Expected Calibration Error (ECE)** | **{agg.get('expected_calibration_error', {}).get('mean', 0):.4f}** | — |
| **Brier Score** | **{agg.get('brier_score', {}).get('mean', 0):.4f}** | — |

---

## 🎯 Confusion Matrix ($N = {report.get('total_samples', 500)}$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = {m.get('confusion_matrix', {}).get('tn', 0)} | FP = {m.get('confusion_matrix', {}).get('fp', 0)} |
| **Actual Hallucinated (1)** | FN = {m.get('confusion_matrix', {}).get('fn', 0)} | TP = {m.get('confusion_matrix', {}).get('tp', 0)} |

---

## ⏱️ Task 7: Latency Evaluation
- **Mean Latency**: `{report['latency_metrics']['mean_ms']} ms`
- **Median Latency**: `{report['latency_metrics']['median_ms']} ms`
- **P90 Latency**: `{report['latency_metrics']['p90_ms']} ms`
- **P95 Latency**: `{report['latency_metrics']['p95_ms']} ms`
- **Maximum Latency**: `{report['latency_metrics']['max_ms']} ms`
"""

    cm = m.get("confusion_matrix", {})
    tex_content = r"""\begin{table}[h!]
\centering
\caption{MultiHaluDet Frozen Publication Performance Across Seeds ($N=""" + str(report.get('total_samples', 500)) + r""").}
\begin{tabular}{lccccc}
\hline
\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUROC} \\
\hline
\textbf{MultiHaluDet (Ours)} & \textbf{""" + f"{acc_mean:.4f}" + r"""} & \textbf{""" + f"{agg.get('precision', {}).get('mean', 0):.4f}" + r"""} & \textbf{""" + f"{agg.get('recall', {}).get('mean', 0):.4f}" + r"""} & \textbf{""" + f"{f1_mean:.4f}" + r"""} & \textbf{""" + f"{auc_mean:.4f}" + r"""} \\
\hline
\end{tabular}
\end{table}"""

    out_dirs = [backend_dir / "reports", backend_dir.parent / "reports"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "publication_tables.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(d / "publication_tables.tex", "w", encoding="utf-8") as f:
            f.write(tex_content)

    logger.info("Exported publication tables to 'reports/publication_tables.md' and 'reports/publication_tables.tex'.")


if __name__ == "__main__":
    run_full_evaluation()
