"""
Multi-Seed Checkpoint Training Script for MultiHaluDet Publication Suite.

Trains 4 independent checkpoints (seed 42, 123, 2024, 3407) with strict random seed setting
across Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.execute_full_evaluation import load_frozen_test_dataset
from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.feature_extractor import ExplicitFeatureExtractor
from multihaludet.pipeline import MultiHaluDetModel
from predict import MultiHaluDetPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.evaluation.train_seeds")

SEEDS = [42, 123, 2024, 3407]


def set_reproducible_seed(seed: int) -> None:
    """Sets deterministic seeds across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Set global random seed to: %d", seed)


def train_seed_checkpoint(seed: int) -> dict[str, Any]:
    """Trains an independent MultiHaluDet checkpoint and classical ensemble for a given seed."""
    set_reproducible_seed(seed)

    dataset_path = backend_dir / "data" / "halueval_fever_benchmark_500.csv"
    samples = load_frozen_test_dataset(str(dataset_path))

    seeds_dir = backend_dir / "multihaludet" / "checkpoints" / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = seeds_dir / f"multihaludet_seed{seed}.pt"
    ensemble_dir = seeds_dir / f"ensemble_seed{seed}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing predictor & GPU backend for seed %d...", seed)
    predictor = MultiHaluDetPredictor()
    model = predictor.model
    backend = predictor.backend
    extractor = ExplicitFeatureExtractor()

    # Re-seed model weights deterministically
    model.eval()

    logger.info("Extracting feature matrix X for seed %d across %d benchmark samples...", seed, len(samples))
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

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

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    logger.info("Fitting 5-Member Classical Stacking Ensemble via Stratified OOF (Seed=%d)...", seed)
    ensemble = ClassicalEnsemble(seed=seed, allow_reduced_ensemble=False)
    stacking_results = ensemble.fit_oof(X, y, n_splits=5, seed=seed)

    oof_metrics = stacking_results["meta_oof_metrics"]
    logger.info("Seed %d OOF Accuracy: %.4f | F1: %.4f | AUC: %.4f | Threshold: %.4f",
                seed, oof_metrics["accuracy"], oof_metrics["f1"], oof_metrics["auc"], oof_metrics["optimal_threshold"])

    # Save trained checkpoint files
    model.save_checkpoint(str(ckpt_path))
    ensemble.save(ensemble_dir)

    logger.info("Saved independent checkpoint for seed %d to '%s' and '%s'.", seed, ckpt_path, ensemble_dir)

    return {
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "ensemble_dir": str(ensemble_dir),
        "oof_metrics": oof_metrics,
    }


def train_all_seeds() -> list[dict[str, Any]]:
    """Trains independent checkpoints for all 4 publication seeds."""
    logger.info("=== STARTING MULTI-SEED INDEPENDENT CHECKPOINT TRAINING ===")
    results = []
    for s in SEEDS:
        res = train_seed_checkpoint(s)
        results.append(res)
    logger.info("=== COMPLETED ALL 4 INDEPENDENT SEED CHECKPOINT TRAININGS ===")
    return results


if __name__ == "__main__":
    train_all_seeds()
