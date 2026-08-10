"""
Trains the MultiHaluDet branch end-to-end (multi-scale attention ->
layer-weighted Transformer -> self-attention pooling -> global branch ->
gated fusion -> classical base-paper stacking ensemble) with out-of-fold feature
generation and reproducible metadata tracking.

Usage:
    python -m multihaludet.training.train \
        --halueval-qa /path/to/halueval_qa.jsonl \
        --triviaqa /path/to/triviaqa_labeled.jsonl \
        --folds 5 --epochs 10 \
        --checkpoint-out ./multihaludet/checkpoints/multihaludet.pt
"""

from __future__ import annotations

import argparse
import datetime
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.generation_backend import GenerationBundle, HFGenerationBackend
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.training.datasets import (
    HallucinationExample,
    get_dataset_diagnostics,
    load_halueval,
    load_multilingual,
    load_triviaqa,
    sample_representative_subset,
)

logger = logging.getLogger("hallucination_guard.multihaludet.train")


def _default_dataset_path(filename: str) -> str | None:
    p = Path("./multihaludet/data") / filename
    return str(p) if p.exists() else None


def _collect_examples(args: argparse.Namespace) -> list[HallucinationExample]:
    examples: list[HallucinationExample] = []
    if args.halueval_qa:
        examples += list(load_halueval(args.halueval_qa, task="qa"))[:2000]
    if args.halueval_dialogue:
        examples += list(load_halueval(args.halueval_dialogue, task="dialogue"))[:2000]
    if args.halueval_summarization:
        examples += list(load_halueval(args.halueval_summarization, task="summarization"))[:2000]
    if args.triviaqa:
        examples += list(load_triviaqa(args.triviaqa))
    if args.french:
        examples += list(load_multilingual(args.french, "fr"))
    if args.bangla:
        examples += list(load_multilingual(args.bangla, "bn"))
    if args.amharic:
        examples += list(load_multilingual(args.amharic, "am"))

    if not examples:
        raise ValueError(
            "No dataset paths given - pass at least one of --halueval-qa / "
            "--halueval-dialogue / --halueval-summarization / --triviaqa / "
            "--french / --bangla / --amharic."
        )

    # Perform representative stratified sampling across (source, label) pairs
    seed = getattr(args, "seed", 42)
    max_samples = getattr(args, "max_samples", None)
    examples = sample_representative_subset(examples, max_samples, seed=seed)
    return examples


def _score_example(backend: HFGenerationBackend, example: HallucinationExample):
    return backend.score_existing_response(example.query, example.response)


def precompute_generation_bundles(
    backend: HFGenerationBackend,
    examples: list[HallucinationExample],
    num_sampled_layers: int = 6,
    top_k_logits: int = 64,
) -> Dict[int, GenerationBundle]:
    """Run the expensive Qwen teacher-forced forward pass exactly once per example.

    The returned GenerationBundles contain compact, memory-efficient Qwen outputs.
    Full float32 vocab logits for 2000 examples consume >120 GB RAM and trigger ArrayMemoryError.
    Top-64 logits + float16 + sampled layers reduce cache RAM from 120GB to ~2.1GB.
    """
    from multihaludet.layer_sampling import select_layers

    cache: Dict[int, GenerationBundle] = {}
    total = len(examples)
    logger.info("=== PRECOMPUTING QWEN GENERATION BUNDLES (%d examples) ===", total)

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(examples), total=total, desc="Precomputing Qwen Bundles", unit="ex")
    except ImportError:
        iterator = enumerate(examples)

    for i, ex in iterator:
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info("Progress: %d/%d Qwen generation bundles precomputed (%.1f%%)", i + 1, total, ((i + 1) / total) * 100)

        bundle = backend.score_existing_response(
            ex.query,
            ex.response,
        )

        if not bundle.is_empty():
            # 1. Convert layer_step_hidden to float16 and sample layers to save RAM
            h = bundle.layer_step_hidden
            if hasattr(h, "detach"):
                h = h.detach().cpu().numpy()
            h = h.astype(np.float16)

            if h.shape[0] > num_sampled_layers:
                sampled_idx = select_layers(h.shape[0], num_sampled_layers)
                h = h[sampled_idx, :, :]

            bundle.layer_step_hidden = h

            # 2. Extract top-64 logits per token step in float16
            logits_arr = bundle.step_logits
            if hasattr(logits_arr, "detach"):
                logits_arr = logits_arr.detach().cpu().numpy()

            if len(logits_arr.shape) == 2 and logits_arr.shape[1] > top_k_logits:
                k = min(top_k_logits, logits_arr.shape[1])
                topk_idx = np.argpartition(logits_arr, -k, axis=-1)[:, -k:]
                rows = np.arange(logits_arr.shape[0])[:, None]
                sorted_sub_idx = np.argsort(-logits_arr[rows, topk_idx], axis=-1)
                logits_arr = logits_arr[rows, topk_idx[rows, sorted_sub_idx]]

            bundle.step_logits = logits_arr.astype(np.float16)

            # 3. Preserve precomputed exact full-vocab step_entropy in float16
            if hasattr(bundle, "step_entropy") and bundle.step_entropy is not None:
                bundle.step_entropy = bundle.step_entropy.astype(np.float16)

        cache[i] = bundle

        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info("Qwen cache progress: %d/%d", i + 1, total)

    logger.info("Qwen preprocessing complete: %d bundles cached in memory", len(cache))
    return cache


def _run_epoch(
    model: MultiHaluDetModel,
    examples: list[HallucinationExample],
    example_indices: list[int],
    cached_bundles: Dict[int, GenerationBundle],
    optimizer: torch.optim.Optimizer,
    epoch_idx: int = 0,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    seen = 0
    order = list(example_indices)
    random.shuffle(order)

    epoch_grad_norms: list[float] = []
    epoch_feature_stds: list[float] = []
    epoch_prob_means: list[float] = []
    epoch_prob_stds: list[float] = []

    for idx in order:
        ex = examples[idx]
        bundle = cached_bundles[idx]
        if bundle.is_empty():
            logger.warning("Skipping empty-response example (idx=%d, source=%s)", idx, ex.source)
            continue

        fused = model.compute_deep_features(bundle)
        out = model.predict_from_features(fused)
        meta_logit = out["meta_logit"].reshape(-1)
        prob = torch.sigmoid(meta_logit)
        target = torch.tensor([1.0 if ex.label else 0.0], dtype=meta_logit.dtype, device=meta_logit.device)

        loss = criterion(meta_logit, target)
        optimizer.zero_grad()
        loss.backward()

        # Step 1: Gradient norm verification across trainable feature extractor & head
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                total_norm += param.grad.norm().item()

        optimizer.step()

        f_mean, f_std = float(fused.mean().item()), (float(fused.std().item()) if fused.numel() > 1 else 0.0)
        f_min, f_max = float(fused.min().item()), float(fused.max().item())
        p_mean = float(prob.mean().item())
        p_std = float(prob.std().item()) if prob.numel() > 1 else 0.0

        epoch_grad_norms.append(total_norm)
        epoch_feature_stds.append(f_std)
        epoch_prob_means.append(p_mean)
        epoch_prob_stds.append(p_std)

        # Log detailed step stats for the first 5 examples of Epoch 1
        if epoch_idx == 0 and seen < 5:
            logger.info(
                "Step %d Diagnostics | GradNorm: %.6f | Feat mean %.4f std %.4f min %.4f max %.4f | Logit %.4f | Prob %.4f",
                seen + 1,
                total_norm,
                f_mean,
                f_std,
                f_min,
                f_max,
                float(meta_logit.mean().item()),
                p_mean,
            )

        total_loss += float(loss.item())
        seen += 1

    if seen > 0:
        avg_norm = float(np.mean(epoch_grad_norms))
        avg_f_std = float(np.mean(epoch_feature_stds))
        avg_p_mean = float(np.mean(epoch_prob_means))
        epoch_p_std = float(np.std(epoch_prob_means))
        logger.info(
            "=== EPOCH %d DIAGNOSTICS SUMMARY ===", epoch_idx + 1
        )
        logger.info(
            "  Average Gradient Norm: %.6f", avg_norm
        )
        logger.info(
            "  Feature Std Dev (Variance): %.6f", avg_f_std
        )
        logger.info(
            "  Probability Mean: %.4f | Epoch Probability Std Dev: %.6f", avg_p_mean, epoch_p_std
        )

    return total_loss / max(seen, 1)


@torch.no_grad()
def _evaluate_examples(
    model: MultiHaluDetModel,
    examples: list[HallucinationExample],
    example_indices: list[int],
    cached_bundles: Dict[int, GenerationBundle],
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []

    for idx in example_indices:
        ex = examples[idx]
        bundle = cached_bundles[idx]
        if bundle.is_empty():
            continue
        fused = model.compute_deep_features(bundle)
        out = model.predict_from_features(fused)
        y_true.append(1 if ex.label else 0)
        y_prob.append(float(out["final_probability"].item()))

    if not y_true:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "auc": float("nan"), "n": 0}

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        "n": len(y_true),
    }


def run_feature_diagnostics(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Computes feature-separation statistics and fits a LogisticRegression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    faithful_mask = (labels == 0)
    halluc_mask = (labels == 1)

    faithful_feats = features[faithful_mask]
    halluc_feats = features[halluc_mask]

    faithful_mean = float(faithful_feats.mean()) if len(faithful_feats) > 0 else 0.0
    faithful_std = float(faithful_feats.std()) if len(faithful_feats) > 0 else 0.0
    halluc_mean = float(halluc_feats.mean()) if len(halluc_feats) > 0 else 0.0
    halluc_std = float(halluc_feats.std()) if len(halluc_feats) > 0 else 0.0

    faithful_l2 = float(np.linalg.norm(faithful_feats, axis=1).mean()) if len(faithful_feats) > 0 else 0.0
    halluc_l2 = float(np.linalg.norm(halluc_feats, axis=1).mean()) if len(halluc_feats) > 0 else 0.0

    faithful_centroid = faithful_feats.mean(axis=0) if len(faithful_feats) > 0 else np.zeros(features.shape[1])
    halluc_centroid = halluc_feats.mean(axis=0) if len(halluc_feats) > 0 else np.zeros(features.shape[1])
    centroid_distance = float(np.linalg.norm(faithful_centroid - halluc_centroid))

    def _avg_cosine(X_a: np.ndarray, X_b: np.ndarray) -> float:
        if len(X_a) == 0 or len(X_b) == 0:
            return 0.0
        norm_a = X_a / (np.linalg.norm(X_a, axis=1, keepdims=True) + 1e-8)
        norm_b = X_b / (np.linalg.norm(X_b, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(norm_a, norm_b.T)
        return float(sim_matrix.mean())

    cosine_within_faithful = _avg_cosine(faithful_feats, faithful_feats)
    cosine_within_halluc = _avg_cosine(halluc_feats, halluc_feats)
    cosine_across = _avg_cosine(faithful_feats, halluc_feats)

    variances = features.var(axis=0)
    near_zero_variance_dims = int(np.sum(variances < 1e-6))

    # Probe model
    probe_metrics = {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}
    if len(set(labels)) > 1 and len(features) >= 4:
        try:
            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(features, labels)
            probs = probe.predict_proba(features)[:, 1]
            preds = (probs >= 0.5).astype(int)
            probe_metrics = {
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "auc": float(roc_auc_score(labels, probs)),
            }
        except Exception as exc:
            logger.warning("Probe fitting error: %s", exc)

    if probe_metrics["auc"] <= 0.52:
        logger.warning(
            "FEATURE DIAGNOSTIC WARNING: Deep features are not currently separating faithful and hallucinated responses (Probe ROC-AUC=%.4f).",
            probe_metrics["auc"],
        )

    return {
        "faithful_feature_mean": faithful_mean,
        "faithful_feature_std": faithful_std,
        "hallucinated_feature_mean": halluc_mean,
        "hallucinated_feature_std": halluc_std,
        "faithful_mean_l2_norm": faithful_l2,
        "hallucinated_mean_l2_norm": halluc_l2,
        "centroid_distance": centroid_distance,
        "avg_cosine_similarity_within_faithful": cosine_within_faithful,
        "avg_cosine_similarity_within_hallucinated": cosine_within_halluc,
        "avg_cosine_similarity_across_classes": cosine_across,
        "overall_feature_variance": float(variances.mean()),
        "near_zero_variance_dimensions": near_zero_variance_dims,
        "probe_metrics": probe_metrics,
    }


def _get_dependency_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "unknown"

    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        versions["scikit-learn"] = "unknown"

    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except Exception:
        versions["xgboost"] = "unavailable"

    try:
        import lightgbm
        versions["lightgbm"] = lightgbm.__version__
    except Exception:
        versions["lightgbm"] = "unavailable"

    return versions


def train(args: argparse.Namespace) -> None:
    from sklearn.model_selection import StratifiedKFold

    examples = _collect_examples(args)
    if getattr(args, "max_samples", None) and len(examples) > args.max_samples:
        import random
        pos_ex = [e for e in examples if e.label]
        neg_ex = [e for e in examples if not e.label]
        half = args.max_samples // 2
        random.seed(args.seed)
        random.shuffle(pos_ex)
        random.shuffle(neg_ex)
        examples = pos_ex[:half] + neg_ex[:half]
        random.seed(args.seed)
        random.shuffle(examples)
        logger.info("Subsampled balanced dataset to %d total examples (%d positive, %d negative)", len(examples), half, half)

    diag_summary = get_dataset_diagnostics(examples)

    logger.info("=== DATASET DIAGNOSTICS ===")
    logger.info("Total examples: %d", diag_summary["total"])
    logger.info("Positive (hallucinated): %d (%.2f%%)", diag_summary["positives"], diag_summary["positive_pct"])
    logger.info("Negative (faithful): %d", diag_summary["negatives"])
    for src, sdata in diag_summary["sources"].items():
        logger.info("  %s: Total=%d, Pos=%d (%.2f%%)", src, sdata["total"], sdata["positives"], sdata["positive_pct"])

    backend = HFGenerationBackend(model_name=args.model_name, device=args.device)

    # Precompute Qwen generation bundles ONCE before fold/epoch loops
    cached_bundles = precompute_generation_bundles(backend, examples)

    # Fold isolation setup
    labels = [1 if ex.label else 0 for ex in examples]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    best_val_auc = -1.0
    best_model: MultiHaluDetModel | None = None
    fold_results: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(examples, labels)):
        logger.info("--- Starting Fold %d/%d ---", fold_idx + 1, args.folds)
        train_idx_list = train_idx.tolist()
        val_idx_list = val_idx.tolist()

        # Step 4: Verify fold label balance
        fold_train_labels = [labels[i] for i in train_idx_list]
        pos_cnt = sum(fold_train_labels)
        neg_cnt = len(fold_train_labels) - pos_cnt
        logger.info("Fold %d Label Balance | Positive (Hallucinated): %d | Negative (Faithful): %d", fold_idx + 1, pos_cnt, neg_cnt)

        # INDEPENDENT MODEL & OPTIMIZER PER FOLD (Prevent Leakage)
        fold_model = MultiHaluDetModel(hidden_size=backend.hidden_size)
        if args.resume_from:
            fold_model.load_checkpoint(args.resume_from)
        fold_optimizer = torch.optim.AdamW(fold_model.parameters(), lr=args.lr, weight_decay=1e-2)

        # Step 5: Verify optimizer parameters on first fold
        if fold_idx == 0:
            trainable_names = [n for n, p in fold_model.named_parameters() if p.requires_grad]
            logger.info("Optimizer tracking %d trainable parameters: %s", len(trainable_names), trainable_names)

        for epoch in range(args.epochs):
            train_loss = _run_epoch(
                fold_model,
                examples,
                train_idx_list,
                cached_bundles,
                fold_optimizer,
                epoch_idx=epoch,
            )
            logger.info(
                "fold %d/%d epoch %d/%d train_loss=%.4f",
                fold_idx + 1, args.folds, epoch + 1, args.epochs, train_loss,
            )

        val_metrics = _evaluate_examples(
            fold_model,
            examples,
            val_idx_list,
            cached_bundles,
        )
        logger.info("fold %d/%d val_metrics=%s", fold_idx + 1, args.folds, val_metrics)
        fold_results.append(val_metrics)

        current_auc = val_metrics.get("auc", 0.5)
        if current_auc > best_val_auc or best_model is None:
            best_val_auc = current_auc
            best_model = fold_model

    assert best_model is not None, "Training failed to produce a valid model"

    # Extract deep features across all training examples for diagnostic & stacking
    logger.info("Extracting deep features for feature-separation diagnostics & OOF stacking...")
    extracted_features: list[np.ndarray] = []
    extracted_labels: list[int] = []

    best_model.eval()
    from multihaludet.feature_extractor import ExplicitFeatureExtractor
    extractor = ExplicitFeatureExtractor()

    with torch.no_grad():
        for idx, ex in enumerate(examples):
            bundle = cached_bundles[idx]
            if bundle.is_empty():
                continue
            fused = best_model.compute_deep_features(bundle)
            fused_np = fused.cpu().numpy().squeeze(0)
            explicit_vec = extractor.extract_feature_vector(ex.query, ex.response)
            combined_vec = np.concatenate([fused_np, explicit_vec], axis=-1)
            extracted_features.append(combined_vec)
            extracted_labels.append(1 if ex.label else 0)

    X_feats = np.array(extracted_features, dtype=np.float32)
    y_labels = np.array(extracted_labels, dtype=np.int64)

    # Run Feature-Separation Diagnostics
    feature_diagnostics = run_feature_diagnostics(X_feats, y_labels)
    logger.info("=== FEATURE SEPARATION DIAGNOSTICS ===")
    for k, v in feature_diagnostics.items():
        logger.info("  %s: %s", k, v)

    # Train Classical Stacking Ensemble (True OOF)
    logger.info("=== TRAINING CLASSICAL STACKING ENSEMBLE (OOF) ===")
    allow_reduced = getattr(args, "allow_reduced_ensemble", False)
    classical_ensemble = ClassicalEnsemble(seed=args.seed, allow_reduced_ensemble=allow_reduced)
    stacking_results = classical_ensemble.fit_oof(X_feats, y_labels, n_splits=args.folds, seed=args.seed)

    best_model.classical_ensemble = classical_ensemble

    # Prediction probability diagnostics
    eval_probs = classical_ensemble.predict_proba(X_feats)["final_probability"]
    eval_probs_arr = np.asarray(eval_probs, dtype=np.float32)
    prob_stats = {
        "min": float(eval_probs_arr.min()),
        "max": float(eval_probs_arr.max()),
        "mean": float(eval_probs_arr.mean()),
        "std": float(eval_probs_arr.std()),
        "percentage_predicted_positive": float((eval_probs_arr >= 0.5).mean() * 100),
    }

    logger.info("=== FINAL ENSEMBLE PROBABILITY DIAGNOSTICS ===")
    logger.info("Min: %.4f, Max: %.4f, Mean: %.4f, Std: %.4f, Pos %%: %.2f%%",
                prob_stats["min"], prob_stats["max"], prob_stats["mean"], prob_stats["std"], prob_stats["percentage_predicted_positive"])

    # Build comprehensive metadata for reproducibility
    metadata: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": args.model_name,
        "sampled_layers": best_model.num_sampled_layers,
        "num_examples_total": len(examples),
        "dataset_diagnostics": diag_summary,
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.lr,
        "feature_dimension": best_model.encoder_dim,
        "base_learner_names": classical_ensemble.active_member_names,
        "is_complete_ensemble": classical_ensemble.is_complete_ensemble,
        "ensemble_mode": classical_ensemble.mode,
        "dependency_versions": _get_dependency_versions(),
        "feature_diagnostics": feature_diagnostics,
        "probability_diagnostics": prob_stats,
        "fold_validation_metrics": fold_results,
        "oof_base_learner_metrics": stacking_results["base_oof_metrics"],
        "oof_meta_learner_metrics": stacking_results["meta_oof_metrics"],
    }

    best_model.save_checkpoint(args.checkpoint_out, metadata=metadata)
    logger.info("Saved trained MultiHaluDet model & complete ensemble artifacts to %s", args.checkpoint_out)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--halueval-qa", default=_default_dataset_path("halueval_qa.jsonl"))
    p.add_argument("--halueval-dialogue", default=_default_dataset_path("halueval_dialogue.jsonl"))
    p.add_argument("--halueval-summarization", default=_default_dataset_path("halueval_summarization.jsonl"))
    p.add_argument("--triviaqa", default=_default_dataset_path("triviaqa_labeled.jsonl"))
    p.add_argument("--french")
    p.add_argument("--bangla")
    p.add_argument("--amharic")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--checkpoint-out", default="./multihaludet/checkpoints/multihaludet.pt")
    p.add_argument("--max-samples", type=int, default=None, help="Maximum number of total samples to use for training (e.g. 2000).")
    p.add_argument("--allow-reduced-ensemble", action="store_true", help="Allow training a reduced ensemble in development/test mode if LightGBM/XGBoost is unavailable.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(_build_arg_parser().parse_args())
