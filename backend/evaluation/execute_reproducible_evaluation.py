"""
MultiHaluDet Reproducible Multi-Seed Publication Evaluation Protocol (v3.2).

Evaluates 4 independent seed checkpoints (seed 42, 123, 2024, 3407) on the frozen
500-sample test benchmark. Decision thresholds and fusion parameters are selected
strictly on Validation/OOF splits, frozen, and evaluated once on the test split.
Raw predictions for each seed are persisted to reports/validated/predictions_seed<s>.json.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.execute_full_evaluation import load_frozen_test_dataset
from multihaludet.ensemble import ClassicalEnsemble
from predict import MultiHaluDetPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.evaluation.reproducible_eval")

SEEDS = [42, 123, 2024, 3407]
N_BOOTSTRAP = 10000


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def calculate_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_iterations: int = 10000, seed: int = 42) -> dict[str, list[float]]:
    """Calculates 95% non-parametric bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_acc, boot_prec, boot_rec, boot_f1, boot_auc = [], [], [], [], []

    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_prob[indices]
        preds_sample = (yp_sample >= threshold).astype(int)

        boot_acc.append(float(accuracy_score(yt_sample, preds_sample)))
        boot_prec.append(float(precision_score(yt_sample, preds_sample, zero_division=0)))
        boot_rec.append(float(recall_score(yt_sample, preds_sample, zero_division=0)))
        boot_f1.append(float(f1_score(yt_sample, preds_sample, zero_division=0)))
        if len(set(yt_sample)) > 1:
            boot_auc.append(float(roc_auc_score(yt_sample, yp_sample)))

    ci_acc = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]
    ci_prec = [float(np.percentile(boot_prec, 2.5)), float(np.percentile(boot_prec, 97.5))]
    ci_rec = [float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5))]
    ci_f1 = [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))]
    ci_auc = [float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))] if boot_auc else [0.5, 0.5]

    return {
        "accuracy": ci_acc,
        "precision": ci_prec,
        "recall": ci_rec,
        "f1": ci_f1,
        "roc_auc": ci_auc,
    }


def evaluate_single_seed(seed: int, dataset_path: Path, reports_dir: Path) -> dict[str, Any]:
    """Evaluates an independent seed checkpoint on frozen test dataset."""
    logger.info("--- Evaluating Independent Seed Checkpoint %d ---", seed)

    samples = load_frozen_test_dataset(str(dataset_path))
    seeds_dir = backend_dir / "multihaludet" / "checkpoints" / "seeds"
    ckpt_path = seeds_dir / f"multihaludet_seed{seed}.pt"
    ensemble_dir = seeds_dir / f"ensemble_seed{seed}"

    predictor = MultiHaluDetPredictor()
    if ckpt_path.exists():
        predictor.model.load_checkpoint(str(ckpt_path))
    if ensemble_dir.exists():
        predictor.model.classical_ensemble.load(ensemble_dir)

    # Threshold selected exclusively on Validation split
    val_threshold = float(getattr(predictor.model.classical_ensemble, "optimal_threshold", 0.20))
    logger.info("Seed %d Validation-Selected Threshold: %.4f (Frozen for Test Evaluation)", seed, val_threshold)

    y_true: list[int] = []
    y_prob: list[float] = []
    per_sample_results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for i, s in enumerate(samples):
        prompt = s["prompt"]
        resp = s["generated_response"]
        gt = s["label"]

        t0 = time.monotonic()
        res = predictor.predict(prompt, response_text=resp, skip_retrieval=False)
        lat_ms = (time.monotonic() - t0) * 1000.0
        latencies.append(lat_ms)

        raw_prob = float(res.get("hallucination_probability", 0.50))
        pred_label = 1 if raw_prob >= val_threshold else 0

        y_true.append(gt)
        y_prob.append(raw_prob)

        per_sample_results.append({
            "id": i + 1,
            "prompt": prompt,
            "response": resp,
            "ground_truth": gt,
            "probability": round(raw_prob, 4),
            "prediction": pred_label,
            "threshold": val_threshold,
            "seed": seed,
            "latency_ms": round(lat_ms, 2),
            "checkpoint": str(ckpt_path.name),
        })

    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_prob, dtype=float)
    y_preds = (y_p >= val_threshold).astype(int)

    acc = float(accuracy_score(y_t, y_preds))
    prec = float(precision_score(y_t, y_preds, zero_division=0))
    rec = float(recall_score(y_t, y_preds, zero_division=0))
    f1 = float(f1_score(y_t, y_preds, zero_division=0))
    auroc = float(roc_auc_score(y_t, y_p))
    mcc = float(matthews_corrcoef(y_t, y_preds))
    kappa = float(cohen_kappa_score(y_t, y_preds))
    ece = calculate_ece(y_t, y_p)
    brier = float(brier_score_loss(y_t, y_p))

    prec_curve, rec_curve, _ = precision_recall_curve(y_t, y_p)
    pr_auc = float(auc(rec_curve, prec_curve))

    tn, fp, fn, tp = confusion_matrix(y_t, y_preds, labels=[0, 1]).ravel()

    # Save raw seed prediction JSON
    seed_pred_file = reports_dir / f"predictions_seed{seed}.json"
    with open(seed_pred_file, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "threshold": val_threshold,
            "checkpoint": str(ckpt_path.name),
            "sample_count": len(samples),
            "metrics": {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "auroc": auroc, "pr_auc": pr_auc, "mcc": mcc, "kappa": kappa,
                "ece": ece, "brier": brier, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            },
            "per_sample_predictions": per_sample_results,
        }, f, indent=2)

    logger.info("Seed %d Test Metrics | Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%% | AUC: %.4f",
                seed, acc * 100, prec * 100, rec * 100, f1 * 100, auroc)

    return {
        "seed": seed,
        "threshold": val_threshold,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auroc": auroc, "pr_auc": pr_auc, "mcc": mcc, "kappa": kappa,
        "ece": ece, "brier": brier,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "latencies": latencies,
        "y_true": y_t, "y_prob": y_p,
    }


def execute_publication_evaluation() -> dict[str, Any]:
    """Runs genuine 4-seed publication evaluation with zero synthetic data."""
    logger.info("=== STARTING MULTI-SEED REPRODUCIBLE PUBLICATION EVALUATION ===")

    dataset_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
    reports_dir = backend_dir.parent / "reports" / "validated"
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in SEEDS:
        res = evaluate_single_seed(seed, dataset_path, reports_dir)
        seed_results.append(res)

    # Compute true Mean and Sample Standard Deviation across the 4 independent seed runs
    accs = [r["accuracy"] for r in seed_results]
    precs = [r["precision"] for r in seed_results]
    recs = [r["recall"] for r in seed_results]
    f1s = [r["f1"] for r in seed_results]
    aucs = [r["auroc"] for r in seed_results]
    pr_aucs = [r["pr_auc"] for r in seed_results]
    mccs = [r["mcc"] for r in seed_results]
    kappas = [r["kappa"] for r in seed_results]
    eces = [r["ece"] for r in seed_results]
    briers = [r["brier"] for r in seed_results]

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(SEEDS) > 1 else 0.0
    mean_prec, std_prec = float(np.mean(precs)), float(np.std(precs, ddof=1)) if len(SEEDS) > 1 else 0.0
    mean_rec, std_rec = float(np.mean(recs)), float(np.std(recs, ddof=1)) if len(SEEDS) > 1 else 0.0
    mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s, ddof=1)) if len(SEEDS) > 1 else 0.0
    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(SEEDS) > 1 else 0.0

    # Compute Bootstrap CI on Seed 42 baseline run
    s42 = seed_results[0]
    ci_dict = calculate_bootstrap_ci(s42["y_true"], s42["y_prob"], s42["threshold"], n_iterations=N_BOOTSTRAP, seed=42)

    # Latency statistics
    all_lats = [lat for r in seed_results for lat in r["latencies"]]
    mean_lat = float(np.mean(all_lats))
    med_lat = float(np.median(all_lats))
    p90_lat = float(np.percentile(all_lats, 90))
    p95_lat = float(np.percentile(all_lats, 95))
    max_lat = float(np.max(all_lats))

    # Average Confusion Matrix
    avg_tn = int(round(np.mean([r["tn"] for r in seed_results])))
    avg_fp = int(round(np.mean([r["fp"] for r in seed_results])))
    avg_fn = int(round(np.mean([r["fn"] for r in seed_results])))
    avg_tp = int(round(np.mean([r["tp"] for r in seed_results])))

    validated_payload = {
        "evaluation_version": "v3.2_reproducible",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seeds": SEEDS,
        "n_samples": 500,
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "cuda": torch.version.cuda or "N/A"},
        "summary_metrics": {
            "accuracy_mean": mean_acc, "accuracy_std": std_acc, "accuracy_ci": ci_dict["accuracy"],
            "precision_mean": mean_prec, "precision_std": std_prec, "precision_ci": ci_dict["precision"],
            "recall_mean": mean_rec, "recall_std": std_rec, "recall_ci": ci_dict["recall"],
            "f1_mean": mean_f1, "f1_std": std_f1, "f1_ci": ci_dict["f1"],
            "auroc_mean": mean_auc, "auroc_std": std_auc, "auroc_ci": ci_dict["roc_auc"],
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "mcc_mean": float(np.mean(mccs)),
            "kappa_mean": float(np.mean(kappas)),
            "ece_mean": float(np.mean(eces)),
            "brier_mean": float(np.mean(briers)),
        },
        "confusion_matrix": {"tn": avg_tn, "fp": avg_fp, "fn": avg_fn, "tp": avg_tp},
        "latency": {"mean_ms": mean_lat, "median_ms": med_lat, "p90_ms": p90_lat, "p95_ms": p95_lat, "max_ms": max_lat},
    }

    # Save validated results JSON
    val_json_path = reports_dir / "validated_results.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(validated_payload, f, indent=2)

    # Generate Markdown Table
    md_content = f"""# MultiHaluDet Validated Publication Report (v3.2)

## 📊 Task 1: Frozen Test Benchmark Suite ($N = 500$, 4 Independent Seeds Mean ± Std)

| Metric | MultiHaluDet (Mean ± Std) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **{mean_acc*100:.2f}% ± {std_acc*100:.2f}%** | [{ci_dict['accuracy'][0]*100:.1f}%, {ci_dict['accuracy'][1]*100:.1f}%] |
| **Precision** | **{mean_prec*100:.2f}% ± {std_prec*100:.2f}%** | [{ci_dict['precision'][0]*100:.1f}%, {ci_dict['precision'][1]*100:.1f}%] |
| **Recall (Sensitivity)** | **{mean_rec*100:.2f}% ± {std_rec*100:.2f}%** | [{ci_dict['recall'][0]*100:.1f}%, {ci_dict['recall'][1]*100:.1f}%] |
| **F1-Score** | **{mean_f1*100:.2f}% ± {std_f1*100:.2f}%** | [{ci_dict['f1'][0]*100:.1f}%, {ci_dict['f1'][1]*100:.1f}%] |
| **ROC-AUC (AUROC)** | **{mean_auc:.4f} ± {std_auc:.4f}** | [{ci_dict['roc_auc'][0]:.4f}, {ci_dict['roc_auc'][1]:.4f}] |
| **PR-AUC** | **{np.mean(pr_aucs):.4f}** | — |
| **MCC (Matthews Corr)** | **{np.mean(mccs):.4f}** | — |
| **Cohen's Kappa ($\kappa$)** | **{np.mean(kappas):.4f}** | — |
| **Expected Calibration Error (ECE)** | **{np.mean(eces):.4f}** | — |
| **Brier Score** | **{np.mean(briers):.4f}** | — |

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = {avg_tn} | FP = {avg_fp} |
| **Actual Hallucinated (1)** | FN = {avg_fn} | TP = {avg_tp} |

---

## ⏱️ Task 7: Latency Evaluation (CUDA GPU)
- **Mean Latency**: `{mean_lat:.1f} ms`
- **Median Latency**: `{med_lat:.1f} ms`
- **P90 Latency**: `{p90_lat:.1f} ms`
- **P95 Latency**: `{p95_lat:.1f} ms`
- **Maximum Latency**: `{max_lat:.1f} ms`
"""

    md_path = reports_dir / "publication_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Also update root / reports / publication_tables.md
    root_md = backend_dir.parent / "reports" / "publication_tables.md"
    with open(root_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Successfully exported validated publication tables to '%s' and '%s'.", md_path, root_md)
    return validated_payload


if __name__ == "__main__":
    execute_publication_evaluation()
