"""
Refit Production System C Checkpoint on ALL Development Data & Freeze Threshold.

Pipeline Workflow:
1. Load all development dataset examples (HaluEval QA, Summarization, Dialogue, TriviaQA).
2. Extract 15 explicit verification & NLI features across all development examples.
3. Compute 5-Fold Stratified OOF cross-validation predictions on development data to derive
   and freeze the optimal decision threshold (Youden J statistic).
4. Refit System C classical ensemble (RF, XGBoost, LightGBM, LogReg, SVM + MetaLearner) on full development set.
5. Save locked production checkpoint to `multihaludet/checkpoints/system_c_final/`.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.feature_extractor import EXPLICIT_FEATURE_NAMES, ExplicitFeatureExtractor
from multihaludet.training.datasets import load_halueval, load_triviaqa
from multihaludet.training.train import _default_dataset_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.multihaludet.fit_system_c")


def fit_and_freeze_system_c(max_samples: int | None = 2000) -> dict[str, Any]:
    logger.info("=== STEP 1: LOADING ALL DEVELOPMENT DATASET EXAMPLES ===")
    examples = []
    data_dir = backend_dir / "multihaludet" / "data"
    for filename, task in [("halueval_qa.jsonl", "qa"), ("halueval_summarization.jsonl", "summarization"), ("halueval_dialogue.jsonl", "dialogue")]:
        p = data_dir / filename
        if p.exists():
            examples.extend(list(load_halueval(str(p), task)))

    tqa_p = data_dir / "triviaqa_labeled.jsonl"
    if tqa_p.exists():
        examples.extend(list(load_triviaqa(str(tqa_p))))

    assert len(examples) > 0, "No development dataset examples found to train System C!"
    logger.info("Total raw development examples loaded: %d", len(examples))

    if max_samples and len(examples) > max_samples:
        import random
        random.seed(42)
        pos = [ex for ex in examples if ex.label]
        neg = [ex for ex in examples if not ex.label]
        random.shuffle(pos)
        random.shuffle(neg)
        half = max_samples // 2
        examples = pos[:half] + neg[:half]
        random.shuffle(examples)
        logger.info("Subsampled to %d balanced development examples (%d positive, %d negative)", len(examples), len(pos[:half]), len(neg[:half]))

    y_dev = np.array([1 if ex.label else 0 for ex in examples], dtype=np.int64)
    logger.info("Class distribution: Positive=%d (%.1f%%) | Negative=%d (%.1f%%)",
                sum(y_dev == 1), np.mean(y_dev) * 100, sum(y_dev == 0), (1 - np.mean(y_dev)) * 100)

    logger.info("\n=== STEP 2: EXTRACTING 15 EXPLICIT VERIFICATION FEATURES ===")
    extractor = ExplicitFeatureExtractor(strict_nli=False)
    explicit_feats = []
    for i, ex in enumerate(examples):
        ev_texts = getattr(ex, "evidence_texts", None)
        vec = extractor.extract_feature_vector(ex.query, ex.response, evidence_texts=ev_texts)
        explicit_feats.append(vec)
        if (i + 1) % 100 == 0 or (i + 1) == len(examples):
            logger.info("Extracted features for %d/%d examples...", i + 1, len(examples))

    X_dev = np.array(explicit_feats, dtype=np.float32)
    assert X_dev.shape[1] == 15, f"Expected 15 features for System C, got {X_dev.shape[1]}"
    logger.info("Explicit feature matrix X_dev shape: %s", X_dev.shape)

    logger.info("\n=== STEP 3: DEVELOPMENT 5-FOLD OOF THRESHOLD FREEZING ===")
    ensemble_oof = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True, system_name="System_C_NLI_Plus_Evidence", expected_feature_dim=15)
    oof_results = ensemble_oof.fit_oof(X_dev, y_dev, n_splits=5, seed=42)

    oof_metrics = oof_results["meta_oof_metrics"]
    frozen_threshold = float(oof_results.get("optimal_threshold", 0.50))

    dev_auc = float(oof_metrics["auc"])
    dev_acc = float(oof_metrics["accuracy"])
    dev_f1 = float(oof_metrics["f1"])
    dev_prec = float(oof_metrics["precision"])
    dev_rec = float(oof_metrics["recall"])

    logger.info("=== DEVELOPMENT OOF METRICS (SYSTEM C) ===")
    logger.info("  Frozen Threshold (Youden's J): %.4f", frozen_threshold)
    logger.info("  Development OOF AUROC       : %.4f (%.2f%%)", dev_auc, dev_auc * 100.0)
    logger.info("  Development OOF Accuracy    : %.4f (%.2f%%)", dev_acc, dev_acc * 100.0)
    logger.info("  Development OOF F1 Score    : %.4f (%.2f%%)", dev_f1, dev_f1 * 100.0)
    logger.info("  Development OOF Precision   : %.4f (%.2f%%)", dev_prec, dev_prec * 100.0)
    logger.info("  Development OOF Recall      : %.4f (%.2f%%)", dev_rec, dev_rec * 100.0)

    logger.info("\n=== STEP 4: REFITTING SYSTEM C ON ALL DEVELOPMENT DATA ===")
    ensemble_final = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True, system_name="System_C_NLI_Plus_Evidence", expected_feature_dim=15)
    ensemble_final.fit_oof(X_dev, y_dev, n_splits=5, seed=42)
    ensemble_final.optimal_threshold = frozen_threshold

    logger.info("\n=== STEP 5: SAVING LOCKED SYSTEM C PRODUCTION CHECKPOINT ===")
    ckpt_dir = backend_dir / "multihaludet" / "checkpoints" / "system_c_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ensemble_final.save(ckpt_dir)

    metadata = {
        "system_name": "System_C_NLI_Plus_Evidence",
        "feature_dim": 15,
        "explicit_feature_names": EXPLICIT_FEATURE_NAMES,
        "num_dev_examples": len(examples),
        "development_oof_metrics": {
            "auroc": dev_auc,
            "accuracy": dev_acc,
            "f1": dev_f1,
            "precision": dev_prec,
            "recall": dev_rec,
            "frozen_threshold": frozen_threshold,
        },
        "frozen_threshold": frozen_threshold,
        "is_complete_ensemble": ensemble_final.is_complete_ensemble,
        "active_member_names": ensemble_final.active_member_names,
    }
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("System C checkpoint locked & saved to '%s'", ckpt_dir)
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit System C Ensemble on Development Data")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max development samples to use (default: 2000)")
    args = parser.parse_args()
    fit_and_freeze_system_c(max_samples=args.max_samples)
