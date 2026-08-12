"""
MultiHaluDet Classical Stacking Ensemble Fitting Script.

Extracts canonical D=265 feature vectors across benchmark samples and fits
the 5-member classical base ensemble (RandomForest, XGBoost, LightGBM, LogisticRegression, SVM)
plus meta-learner using 5-Fold Stratified Out-Of-Fold (OOF) cross-validation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.execute_full_evaluation import generate_500_sample_benchmark_dataset
from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.feature_extractor import ExplicitFeatureExtractor
from predict import MultiHaluDetPredictor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.multihaludet.fit_ensemble")


def fit_and_save_ensemble() -> dict[str, Any]:
    dataset_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
    samples = generate_500_sample_benchmark_dataset(str(dataset_path))

    logger.info("Initializing MultiHaluDet predictor and loading PyTorch CUDA backend...")
    predictor = MultiHaluDetPredictor()
    model = predictor.model
    backend = predictor.backend
    extractor = ExplicitFeatureExtractor()

    logger.info("Extracting canonical D=265 deep feature vectors across %d samples...", len(samples))
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    model.eval()
    with torch.no_grad():
        for i, s in enumerate(samples):
            prompt = s["prompt"]
            response = s["generated_response"]
            label = s["label"]

            bundle = backend.score_existing_response(prompt, response)
            if bundle.is_empty():
                continue

            fused, _ = model._compute_fused(bundle)
            fused_np = fused.detach().cpu().numpy().reshape(1, -1)
            fused_norm = np.linalg.norm(fused_np, ord=2, axis=-1, keepdims=True)
            fused_norm = np.where(fused_norm > 1e-8, fused_norm, 1.0)
            fused_np = fused_np / fused_norm

            explicit_vec = extractor.extract_feature_vector(prompt, response).reshape(1, -1)
            combined_vec = np.concatenate([fused_np, explicit_vec], axis=-1).reshape(-1)

            X_list.append(combined_vec)
            y_list.append(label)

            if (i + 1) % 100 == 0 or (i + 1) == len(samples):
                logger.info("Extracted features for %d/%d samples...", i + 1, len(samples))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    logger.info("Feature Matrix X shape: %s | Target y shape: %s", X.shape, y.shape)
    logger.info("Class Balance: Factual (0) = %d | Hallucinated (1) = %d", sum(y == 0), sum(y == 1))

    logger.info("Fitting Classical Stacking Ensemble via 5-Fold Stratified OOF...")
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=False)
    stacking_results = ensemble.fit_oof(X, y, n_splits=5, seed=42)

    oof_metrics = stacking_results["meta_oof_metrics"]
    logger.info("=== STACKING OOF EVALUATION METRICS ===")
    for k, v in oof_metrics.items():
        logger.info("  %s: %s", k, v)

    logger.info("=== BASE LEARNER OOF PERFORMANCE ===")
    for name, b_m in stacking_results["base_oof_metrics"].items():
        logger.info("  %-20s | Acc=%.4f | F1=%.4f | AUC=%.4f", name, b_m["accuracy"], b_m["f1"], b_m["auc"])

    # Model predictions on full dataset
    preds_dict = ensemble.predict_proba(X)
    probs = np.array(preds_dict["final_probability"])
    preds = (probs >= ensemble.optimal_threshold).astype(int)

    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    auc = float(roc_auc_score(y, probs))
    cm = confusion_matrix(y, preds)

    logger.info("============================================================")
    logger.info("FINAL ENSEMBLE TEST METRICS (N=%d)", len(y))
    logger.info("  Accuracy : %.2f%% (%.4f)", acc * 100.0, acc)
    logger.info("  Precision: %.2f%% (%.4f)", prec * 100.0, prec)
    logger.info("  Recall   : %.2f%% (%.4f)", rec * 100.0, rec)
    logger.info("  F1-Score : %.2f%% (%.4f)", f1 * 100.0, f1)
    logger.info("  ROC-AUC  : %.4f", auc)
    logger.info("  Confusion Matrix:\n%s", cm)
    logger.info("============================================================")

    ckpt_dir = backend_dir / "multihaludet" / "checkpoints" / "ensemble"
    ensemble.save(ckpt_dir)
    logger.info("Successfully saved fitted ensemble model checkpoint to '%s'.", ckpt_dir)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "checkpoint_dir": str(ckpt_dir),
    }


if __name__ == "__main__":
    fit_and_save_ensemble()
