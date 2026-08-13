"""
Single-Run Evaluation of Locked System C Production Checkpoint on Frozen 500 Benchmark.

Executes ONE-TIME held-out evaluation for publication metrics on System C:
1. Loads System C ensemble checkpoint from `multihaludet/checkpoints/system_c_final/`.
2. Loads frozen 500-sample test set (data/halueval_fever_benchmark_500.csv or data/halueval_benchmark_500.jsonl).
3. Extracts 15 explicit domain-specific verification & NLI features.
4. Evaluates ensemble predictions against the frozen development-OOF decision threshold.
5. Computes publication-ready classification metrics, 95% bootstrap CIs, ECE, Brier score, and confusion matrix.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc as calc_auc,
)

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.feature_extractor import ExplicitFeatureExtractor
from multihaludet.training.datasets import load_frozen_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.multihaludet.evaluate_system_c_frozen500")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = np.where((y_prob >= bins[i]) & (y_prob < bins[i + 1]))[0]
        if len(idx) > 0:
            acc = float(np.mean(y_true[idx]))
            conf = float(np.mean(y_prob[idx]))
            ece += (len(idx) / len(y_true)) * abs(acc - conf)
    return float(ece)


def compute_bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bootstraps: int = 1000, seed: int = 42
) -> dict[str, tuple[float, float]]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, f1s, aucs, pr_aucs = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_true[idx])) <= 1:
            continue
        preds = (y_prob[idx] >= threshold).astype(int)
        accs.append(accuracy_score(y_true[idx], preds))
        f1s.append(f1_score(y_true[idx], preds, zero_division=0))
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true[idx], y_prob[idx])
        pr_aucs.append(calc_auc(rec_arr, prec_arr))

    def ci(arr):
        return (round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)) if arr else (0.0, 0.0)

    return {
        "accuracy_95ci": ci(accs),
        "f1_95ci": ci(f1s),
        "auc_95ci": ci(aucs),
        "pr_auc_95ci": ci(pr_aucs),
    }


def evaluate_system_c_frozen500(checkpoint_dir: str | None = None, frozen_test_path: str | None = None) -> dict[str, Any]:
    logger.info("=== STARTING ONE-TIME HELD-OUT EVALUATION ON FROZEN 500 BENCHMARK (SYSTEM C) ===")

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else backend_dir / "multihaludet" / "checkpoints" / "system_c_final"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"System C checkpoint directory not found at {ckpt_path}. Run fit_system_c.py first.")

    ensemble = ClassicalEnsemble(allow_reduced_ensemble=True, system_name="System_C_NLI_Plus_Evidence", expected_feature_dim=15)
    if not ensemble.load(ckpt_path):
        raise RuntimeError(f"Failed to load System C ensemble checkpoint from {ckpt_path}")

    meta_file = ckpt_path / "metadata.json"
    frozen_threshold = getattr(ensemble, "optimal_threshold", 0.50)
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            frozen_threshold = float(meta_data.get("frozen_threshold", frozen_threshold))

    logger.info("Loaded System C Checkpoint from '%s'", ckpt_path)
    logger.info("Frozen Decision Threshold (derived strictly from Development OOF): %.4f", frozen_threshold)

    # Resolve Test Set Path
    if frozen_test_path and Path(frozen_test_path).exists():
        test_path = Path(frozen_test_path)
    else:
        test_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
        if not test_path.exists():
            test_path = backend_dir / "multihaludet" / "data" / "halueval_benchmark_500.jsonl"

    logger.info("Loading Frozen Test Dataset from '%s'...", test_path)
    test_examples = load_frozen_benchmark(str(test_path))
    logger.info("Total Frozen Test Examples Loaded: %d", len(test_examples))
    y_true = np.array([1 if ex.label else 0 for ex in test_examples], dtype=np.int64)

    logger.info("Extracting 15 Explicit Verification Features on Frozen Test Set...")
    extractor = ExplicitFeatureExtractor(strict_nli=False)
    explicit_feats = []
    from retrieval.evidence_cache import get_evidence_cache
    evidence_cache = get_evidence_cache()

    for i, ex in enumerate(test_examples):
        ev_texts = getattr(ex, "evidence_texts", None)
        if not ev_texts:
            cached_snips = evidence_cache.get_or_fetch(ex.query, top_k=3)
            ev_texts = [s.get("text", "") for s in cached_snips if s.get("text")]
        vec = extractor.extract_feature_vector(ex.query, ex.response, evidence_texts=ev_texts)
        explicit_feats.append(vec)
        if (i + 1) % 50 == 0 or (i + 1) == len(test_examples):
            logger.info("Extracted System C features for %d/%d test samples...", i + 1, len(test_examples))

    X_test = np.array(explicit_feats, dtype=np.float32)

    logger.info("Running System C Stacking Ensemble Inference...")
    pred_res = ensemble.predict_proba(X_test)
    y_prob = np.array(pred_res["final_probability"], dtype=np.float32)

    y_pred = (y_prob >= frozen_threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(calc_auc(rec_arr, prec_arr))

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    ci_dict = compute_bootstrap_ci(y_true, y_prob, threshold=frozen_threshold)

    # Base Learner Member Metrics
    base_member_metrics = {}
    if "member_probabilities" in pred_res and isinstance(pred_res["member_probabilities"], dict):
        for member_name, member_probs_list in pred_res["member_probabilities"].items():
            m_probs = np.array(member_probs_list, dtype=np.float32)
            m_auc = float(roc_auc_score(y_true, m_probs)) if len(set(y_true)) > 1 else 0.5
            base_member_metrics[member_name] = round(m_auc, 4)

    logger.info("\n================================================================================")
    logger.info("FINAL PUBLICATION METRICS — FROZEN 500 BENCHMARK (SYSTEM C)")
    logger.info("================================================================================")
    logger.info("  Evaluated Test Set Size : %d", len(y_true))
    logger.info("  Decision Threshold      : %.4f (Locked from Dev OOF)", frozen_threshold)
    logger.info("  Accuracy                : %.4f (%.2f%%) | 95%% CI: %s", acc, acc * 100.0, ci_dict["accuracy_95ci"])
    logger.info("  Precision               : %.4f (%.2f%%)", prec, prec * 100.0)
    logger.info("  Recall                  : %.4f (%.2f%%)", rec, rec * 100.0)
    logger.info("  F1-Score                : %.4f (%.2f%%) | 95%% CI: %s", f1, f1 * 100.0, ci_dict["f1_95ci"])
    logger.info("  ROC-AUC                 : %.4f (%.2f%%) | 95%% CI: %s", auc, auc * 100.0, ci_dict["auc_95ci"])
    logger.info("  PR-AUC                  : %.4f (%.2f%%) | 95%% CI: %s", pr_auc, pr_auc * 100.0, ci_dict["pr_auc_95ci"])
    logger.info("  Brier Score             : %.4f", brier)
    logger.info("  ECE (Calibration Error) : %.4f", ece)
    logger.info("  Confusion Matrix        : %s", cm)
    logger.info("  Base Learner Member AUCs: %s", base_member_metrics)
    logger.info("================================================================================\n")

    results = {
        "system_name": "System_C_NLI_Plus_Evidence",
        "checkpoint_dir": str(ckpt_path),
        "test_dataset_path": str(test_path),
        "num_test_samples": len(y_true),
        "frozen_threshold": frozen_threshold,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "expected_calibration_error": ece,
            "confusion_matrix": cm,
            "base_member_aucs": base_member_metrics,
        },
        "bootstrap_ci": ci_dict,
    }

    report_path = backend_dir / "data" / "system_c_frozen_500_publication_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved final publication results JSON report to '%s'", report_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate System C on Frozen 500 Benchmark")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--frozen-test", type=str, default=None)
    args = parser.parse_args()
    evaluate_system_c_frozen500(args.checkpoint_dir, args.frozen_test)


if __name__ == "__main__":
    main()
