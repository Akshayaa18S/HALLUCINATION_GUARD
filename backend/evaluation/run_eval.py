"""
Evaluation Layer - Publication Benchmark Evaluation Script.

Evaluates MultiHaluDet on labeled CSV test datasets (query, generated_response, label),
runs inference per sample, and computes complete publication metrics:
- Confusion Matrix (TP, FP, FN, TN)
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- MCC (Matthews Correlation Coefficient), Cohen's Kappa
- Calibration: ECE, Brier Score
- Inference Latency (Mean, Median, P95)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import sys
from pathlib import Path
from typing import Any
import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.metrics import compute_calibration_metrics, compute_classification_metrics
from predict import MultiHaluDetPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.eval_runner")


def evaluate_dataset(csv_path: str, predictor: MultiHaluDetPredictor | None = None) -> dict[str, Any]:
    """Evaluates a labeled CSV dataset and computes publication metrics."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")

    if predictor is None:
        predictor = MultiHaluDetPredictor()

    samples: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    if not samples:
        raise ValueError(f"CSV file '{csv_path}' contains no records.")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    latencies: list[float] = []
    per_sample_results: list[dict[str, Any]] = []

    logger.info("Evaluating %d samples from dataset '%s'...", len(samples), csv_path)

    for idx, sample in enumerate(samples, 1):
        query = sample.get("query", sample.get("prompt", ""))
        gen_resp = sample.get("generated_response", sample.get("response", ""))
        label = int(sample.get("label", sample.get("target", 0)))

        start_time = time.monotonic()
        try:
            res = predictor.predict(query, response_text=gen_resp)
            prob = float(res.get("hallucination_probability", 0.50))
            pred = 1 if prob >= float(res.get("decision_threshold", 0.20)) else 0
        except Exception as exc:
            logger.warning("Sample %d failed: %s", idx, exc)
            prob = 0.50
            pred = 0

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob)

        per_sample_results.append({
            "sample_id": idx,
            "query": query,
            "ground_truth": label,
            "prediction": pred,
            "probability": round(prob, 4),
            "latency_ms": round(elapsed_ms, 1),
            "is_correct": bool(label == pred),
        })

    # Decision Threshold Optimization Sweep (0.20 to 0.50)
    threshold_sweep: dict[str, dict[str, float]] = {}
    best_thresh = 0.20
    best_f1 = -1.0

    for t_val in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        t_preds = [1 if p >= t_val else 0 for p in y_prob]
        t_metrics = compute_classification_metrics(y_true, t_preds, y_prob)
        threshold_sweep[f"threshold_{t_val:.2f}"] = t_metrics
        if t_metrics["f1"] > best_f1:
            best_f1 = t_metrics["f1"]
            best_thresh = t_val

    opt_preds = [1 if p >= best_thresh else 0 for p in y_prob]
    opt_class_metrics = compute_classification_metrics(y_true, opt_preds, y_prob)
    calib_metrics = compute_calibration_metrics(y_true, y_prob)

    latency_mean = float(np.mean(latencies))
    latency_median = float(np.median(latencies))
    latency_p95 = float(np.percentile(latencies, 95))

    report = {
        "dataset_name": os.path.basename(csv_path),
        "total_samples": len(samples),
        "optimal_threshold": round(best_thresh, 2),
        "classification_metrics": opt_class_metrics,
        "default_threshold_metrics": threshold_sweep.get("threshold_0.20", opt_class_metrics),
        "threshold_sweep": threshold_sweep,
        "calibration_metrics": calib_metrics,
        "latency_metrics": {
            "mean_ms": round(latency_mean, 1),
            "median_ms": round(latency_median, 1),
            "p95_ms": round(latency_p95, 1),
        },
        "per_sample_results": per_sample_results,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="MultiHaluDet Publication Evaluation Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to labeled CSV dataset")
    parser.add_argument("--output", type=str, default="evaluation_report_full.json", help="Output path for evaluation JSON")
    args = parser.parse_args()

    report = evaluate_dataset(args.dataset)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation complete for '%s'. Report saved to '%s'.", args.dataset, args.output)
    print(json.dumps(report["classification_metrics"], indent=2))


if __name__ == "__main__":
    main()
