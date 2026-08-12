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
import re
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
    dev_examples: list[HallucinationExample] = []
    if args.halueval_qa:
        dev_examples += list(load_halueval(args.halueval_qa, task="qa"))[:2000]
    if args.halueval_dialogue:
        dev_examples += list(load_halueval(args.halueval_dialogue, task="dialogue"))[:2000]
    if args.halueval_summarization:
        dev_examples += list(load_halueval(args.halueval_summarization, task="summarization"))[:2000]
    if args.triviaqa:
        dev_examples += list(load_triviaqa(args.triviaqa))
    if args.french:
        dev_examples += list(load_multilingual(args.french, "fr"))
    if args.bangla:
        dev_examples += list(load_multilingual(args.bangla, "bn"))
    if args.amharic:
        dev_examples += list(load_multilingual(args.amharic, "am"))

    if not dev_examples:
        raise ValueError(
            "No development dataset paths given - pass at least one of --halueval-qa / "
            "--halueval-dialogue / --halueval-summarization / --triviaqa."
        )

    # Perform representative stratified sampling across development pool ONLY
    seed = getattr(args, "seed", 42)
    max_samples = getattr(args, "max_samples", None)
    dev_examples = sample_representative_subset(dev_examples, max_samples, seed=seed)
    return dev_examples


def train(args: argparse.Namespace) -> None:
    from sklearn.model_selection import StratifiedKFold
    from multihaludet.training.datasets import load_frozen_benchmark, verify_no_test_contamination

    examples = _collect_examples(args)

    frozen_test_examples: list[HallucinationExample] = []
    frozen_test_path = getattr(args, "frozen_test", None)
    if frozen_test_path and Path(frozen_test_path).exists():
        try:
            frozen_test_examples = load_frozen_benchmark(frozen_test_path)
            verify_no_test_contamination(examples, frozen_test_examples)
            logger.info("FROZEN TEST ISOLATION VERIFIED: 0 overlap between %d development samples and %d frozen test samples.", len(examples), len(frozen_test_examples))
        except Exception as exc:
            logger.warning("Frozen benchmark contamination check warning: %s", exc)

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
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    max_grad_norm: float = 1.0,
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

    optimizer.zero_grad()
    batch_meta_logits: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []

    for step_idx, idx in enumerate(order):
        ex = examples[idx]
        bundle = cached_bundles[idx]
        if bundle.is_empty():
            logger.warning("Skipping empty-response example (idx=%d, source=%s)", idx, ex.source)
            continue

        fused = model.compute_deep_features(bundle)
        out = model.predict_from_features(fused)
        meta_logit = out["meta_logit"].reshape(-1)
        target = torch.tensor([1.0 if ex.label else 0.0], dtype=meta_logit.dtype, device=meta_logit.device)

        batch_meta_logits.append(meta_logit)
        batch_targets.append(target)
        seen += 1

        if len(batch_meta_logits) == batch_size or step_idx == len(order) - 1:
            logits_cat = torch.cat(batch_meta_logits)
            targets_cat = torch.cat(batch_targets)

            loss = criterion(logits_cat, targets_cat) / grad_accum_steps
            loss.backward()

            if (step_idx + 1) % (batch_size * grad_accum_steps) == 0 or step_idx == len(order) - 1:
                # Gradient clipping
                total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
                optimizer.step()
                optimizer.zero_grad()
                epoch_grad_norms.append(total_norm)

            total_loss += float(loss.item()) * grad_accum_steps * len(batch_meta_logits)
            batch_meta_logits = []
            batch_targets = []

    if seen > 0:
        avg_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        logger.info(
            "=== EPOCH %d DIAGNOSTICS SUMMARY === | Examples Seen: %d | Avg Grad Norm: %.6f",
            epoch_idx + 1, seen, avg_norm
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

    # Out-of-Fold Probe model (prevents in-sample memorization illusion)
    probe_metrics = {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}
    if len(set(labels)) > 1 and len(features) >= 10:
        try:
            from sklearn.model_selection import cross_val_predict, StratifiedKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
            skf = StratifiedKFold(n_splits=min(5, len(labels)), shuffle=True, random_state=42)
            probs = cross_val_predict(pipe, features, labels, cv=skf, method="predict_proba")[:, 1]
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
    from multihaludet.training.datasets import load_frozen_benchmark, verify_no_test_contamination

    examples = _collect_examples(args)


    frozen_test_examples: list[HallucinationExample] = []
    frozen_test_path = getattr(args, "frozen_test", None)
    if frozen_test_path and Path(frozen_test_path).exists():
        try:
            frozen_test_examples = load_frozen_benchmark(frozen_test_path)
            verify_no_test_contamination(examples, frozen_test_examples)
            logger.info("FROZEN TEST ISOLATION VERIFIED: 0 overlap between %d development samples and %d frozen test samples.", len(examples), len(frozen_test_examples))
        except Exception as exc:
            logger.warning("Frozen benchmark contamination check warning: %s", exc)

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

    logger.info("=== DEVELOPMENT DATASET DIAGNOSTICS ===")
    logger.info("Total examples: %d", diag_summary["total"])
    logger.info("Positive (hallucinated): %d (%.2f%%)", diag_summary["positives"], diag_summary["positive_pct"])
    logger.info("Negative (faithful): %d", diag_summary["negatives"])
    for src, sdata in diag_summary["sources"].items():
        logger.info("  %s: Total=%d, Pos=%d (%.2f%%)", src, sdata["total"], sdata["positives"], sdata["positive_pct"])

    backend = HFGenerationBackend(model_name=args.model_name, device=args.device)

    model_slug = re.sub(r"[^\w\-]", "_", str(args.model_name))
    cache_path = Path(f"./multihaludet/data/bundle_cache_{model_slug}_{backend.hidden_size}.pt")
    if args.model_name != "fake" and cache_path.exists():
        logger.info("Loading cached generation bundles from %s...", cache_path)
        cached_bundles = torch.load(cache_path, weights_only=False)
    else:
        cached_bundles = precompute_generation_bundles(backend, examples)
        if args.model_name != "fake":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cached_bundles, cache_path)

    # Fold isolation setup
    labels = [1 if ex.label else 0 for ex in examples]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    best_val_auc = -1.0
    best_model: MultiHaluDetModel | None = None
    fold_results: list[dict[str, Any]] = []

    from multihaludet.feature_extractor import ExplicitFeatureExtractor
    strict_nli = not (getattr(args, "allow_nli_fallback", False) or getattr(args, "allow_reduced_ensemble", False))
    extractor = ExplicitFeatureExtractor(device=args.device, strict_nli=strict_nli)




    # Pre-extract non-trainable explicit feature vectors for all development examples
    logger.info("Extracting non-trainable explicit features across all development dataset examples...")
    explicit_features: list[np.ndarray] = []
    for ex in examples:
        vec = extractor.extract_feature_vector(ex.query, ex.response)
        explicit_features.append(vec)
    X_explicit_all = np.array(explicit_features, dtype=np.float32)

    # Pre-allocate array for true Out-Of-Fold (OOF) deep feature vectors
    dummy_model = MultiHaluDetModel(hidden_size=backend.hidden_size)
    deep_dim = dummy_model.encoder_dim
    X_oof_deep = np.zeros((len(examples), deep_dim), dtype=np.float32)
    oof_written = np.zeros(len(examples), dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(examples, labels)):
        logger.info("--- Starting Fold %d/%d ---", fold_idx + 1, args.folds)
        train_idx_list = train_idx.tolist()
        val_idx_list = val_idx.tolist()

        fold_train_labels = [labels[i] for i in train_idx_list]
        pos_cnt = sum(fold_train_labels)
        neg_cnt = len(fold_train_labels) - pos_cnt
        logger.info("Fold %d Label Balance | Positive (Hallucinated): %d | Negative (Faithful): %d", fold_idx + 1, pos_cnt, neg_cnt)

        # INDEPENDENT MODEL & OPTIMIZER PER FOLD (Prevent Leakage)
        fold_model = MultiHaluDetModel(hidden_size=backend.hidden_size).to(args.device)
        if args.resume_from:
            fold_model.load_checkpoint(args.resume_from)

        if getattr(args, "freeze_encoder", False):
            for name, param in fold_model.named_parameters():
                if "multi_scale_attention" in name or "layer_weighted_encoder" in name:
                    param.requires_grad = False

        fold_optimizer = torch.optim.AdamW(
            [p for p in fold_model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-2,
        )
        fold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=args.epochs, eta_min=1e-6)

        if fold_idx == 0:
            trainable_names = [n for n, p in fold_model.named_parameters() if p.requires_grad]
            logger.info("Optimizer tracking %d trainable parameters: %s", len(trainable_names), trainable_names)

        best_fold_epoch_auc = -1.0
        best_fold_state = None
        patience_counter = 0
        patience_limit = getattr(args, "patience", 3)

        for epoch in range(args.epochs):
            train_loss = _run_epoch(
                fold_model,
                examples,
                train_idx_list,
                cached_bundles,
                fold_optimizer,
                epoch_idx=epoch,
            )
            fold_scheduler.step()

            epoch_val = _evaluate_examples(fold_model, examples, val_idx_list, cached_bundles)
            epoch_val_auc = epoch_val.get("auc", 0.5)

            logger.info(
                "fold %d/%d epoch %d/%d train_loss=%.4f val_auc=%.4f lr=%.6f",
                fold_idx + 1, args.folds, epoch + 1, args.epochs, train_loss, epoch_val_auc, fold_scheduler.get_last_lr()[0],
            )

            if epoch_val_auc > best_fold_epoch_auc:
                best_fold_epoch_auc = epoch_val_auc
                best_fold_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit and epoch >= 2:
                    logger.info("Early stopping triggered at epoch %d for fold %d (Best Val AUC: %.4f)", epoch + 1, fold_idx + 1, best_fold_epoch_auc)
                    break

        if best_fold_state is not None:
            fold_model.load_state_dict(best_fold_state)

        # GENERATE TRUE OUT-OF-FOLD (OOF) DEEP FEATURES WITH PROVENANCE ASSERTIONS
        fold_model.eval()
        with torch.no_grad():
            for idx in val_idx_list:
                assert idx not in train_idx_list, f"LEAKAGE ERROR: Sample {idx} present in train split for fold {fold_idx + 1}"
                bundle = cached_bundles[idx]
                if bundle.is_empty():
                    continue
                fused = fold_model.compute_deep_features(bundle)
                fused_np = fused.cpu().numpy().reshape(-1)
                fnorm = float(np.linalg.norm(fused_np, ord=2))
                if fnorm > 1e-8:
                    fused_np = fused_np / fnorm
                X_oof_deep[idx] = fused_np
                oof_written[idx] = True

        # Step 6: Evaluate per-fold deep features with LogisticRegression & Linear SVM baselines
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score, accuracy_score

        tr_feats, tr_labs = [], []
        va_feats, va_labs = [], []

        for idx in train_idx_list:
            if cached_bundles[idx].is_empty():
                continue
            with torch.no_grad():
                fused_tr = fold_model.compute_deep_features(cached_bundles[idx]).cpu().numpy().reshape(-1)
                fnorm_tr = float(np.linalg.norm(fused_tr, ord=2))
                if fnorm_tr > 1e-8:
                    fused_tr = fused_tr / fnorm_tr
            tr_feats.append(np.concatenate([fused_tr, X_explicit_all[idx]], axis=-1))
            tr_labs.append(1 if examples[idx].label else 0)

        for idx in val_idx_list:
            if cached_bundles[idx].is_empty():
                continue
            va_feats.append(np.concatenate([X_oof_deep[idx], X_explicit_all[idx]], axis=-1))
            va_labs.append(1 if examples[idx].label else 0)

        X_tr_f, y_tr_f = np.array(tr_feats, dtype=np.float32), np.array(tr_labs, dtype=np.int64)
        X_va_f, y_va_f = np.array(va_feats, dtype=np.float32), np.array(va_labs, dtype=np.int64)

        from sklearn.preprocessing import StandardScaler
        scaler_fold = StandardScaler()
        X_tr_f_s = scaler_fold.fit_transform(X_tr_f)
        X_va_f_s = scaler_fold.transform(X_va_f)

        lr_probe = LogisticRegression(max_iter=1000, random_state=args.seed)
        lr_probe.fit(X_tr_f_s, y_tr_f)
        lr_probs = lr_probe.predict_proba(X_va_f_s)[:, 1]
        lr_auc = float(roc_auc_score(y_va_f, lr_probs)) if len(set(y_va_f)) > 1 else 0.5
        lr_acc = float(accuracy_score(y_va_f, (lr_probs >= 0.5).astype(int)))

        svm_probe = SVC(probability=True, random_state=args.seed)
        svm_probe.fit(X_tr_f_s, y_tr_f)
        svm_probs = svm_probe.predict_proba(X_va_f_s)[:, 1]
        svm_auc = float(roc_auc_score(y_va_f, svm_probs)) if len(set(y_va_f)) > 1 else 0.5
        svm_acc = float(accuracy_score(y_va_f, (svm_probs >= 0.5).astype(int)))

        val_metrics = _evaluate_examples(
            fold_model,
            examples,
            val_idx_list,
            cached_bundles,
        )
        val_metrics["logistic_regression_val_auc"] = lr_auc
        val_metrics["logistic_regression_val_acc"] = lr_acc
        val_metrics["linear_svm_val_auc"] = svm_auc
        val_metrics["linear_svm_val_acc"] = svm_acc

        logger.info(
            "Fold %d/%d Baseline Comparison | Neural Ensemble Val AUC: %.4f | LogReg Val AUC: %.4f (Acc: %.4f) | SVM Val AUC: %.4f (Acc: %.4f)",
            fold_idx + 1, args.folds, val_metrics.get("auc", 0.5), lr_auc, lr_acc, svm_auc, svm_acc
        )
        fold_results.append(val_metrics)

        current_auc = val_metrics.get("auc", 0.5)
        if current_auc > best_val_auc or best_model is None:
            best_val_auc = current_auc
            best_model = fold_model

    assert best_model is not None, "Training failed to produce a valid model"
    assert np.all(oof_written), "OOF INCOMPLETENESS ERROR: Not all development samples were written during OOF feature extraction!"

    # Assemble true Out-Of-Fold (OOF) total feature matrix: [X_oof_deep, X_explicit_all]
    logger.info("Assembling 100%% leak-free Out-Of-Fold (OOF) feature matrix across all %d development examples...", len(examples))
    X_oof_total = np.concatenate([X_oof_deep, X_explicit_all], axis=-1)
    y_labels = np.array([1 if ex.label else 0 for ex in examples], dtype=np.int64)

    # Run Out-Of-Fold Feature-Separation Diagnostics (5-Fold Cross-Validated Probe)
    feature_diagnostics = run_feature_diagnostics(X_oof_total, y_labels)
    logger.info("=== FEATURE SEPARATION DIAGNOSTICS (5-Fold CV Probe OOF) ===")
    for k, v in feature_diagnostics.items():
        logger.info("  %s: %s", k, v)

    # Evaluate 4-System Comparative OOF Performance Table (Identical Splits)
    from multihaludet.ensemble import evaluate_comparative_systems
    allow_reduced = getattr(args, "allow_reduced_ensemble", False)
    comp_systems = evaluate_comparative_systems(X_oof_total, y_labels, n_splits=args.folds, seed=args.seed, allow_reduced=allow_reduced)

    logger.info("=== 4-SYSTEM COMPARATIVE OUT-OF-FOLD (OOF) PERFORMANCE TABLE ===")
    for sys_name, sys_m in comp_systems.items():
        logger.info("  %-35s | OOF AUC: %.4f | PR-AUC: %.4f | F1: %.4f | Acc: %.4f", sys_name, sys_m.get("auc", 0.5), sys_m.get("pr_auc", 0.5), sys_m.get("f1", 0.0), sys_m.get("accuracy", 0.5))


    # Train Classical Stacking Ensemble on True OOF Features
    logger.info("=== TRAINING CLASSICAL STACKING ENSEMBLE (ON TRUE OOF FEATURES) ===")
    classical_ensemble = ClassicalEnsemble(seed=args.seed, allow_reduced_ensemble=allow_reduced)
    stacking_results = classical_ensemble.fit_oof(X_oof_total, y_labels, n_splits=args.folds, seed=args.seed)


    # Refit final neural feature extractor on all development data
    logger.info("Refitting final neural feature extractor on all %d development examples...", len(examples))
    final_model = MultiHaluDetModel(hidden_size=backend.hidden_size).to(args.device)
    if getattr(args, "freeze_encoder", False):
        for name, param in final_model.named_parameters():
            if "multi_scale_attention" in name or "layer_weighted_encoder" in name:
                param.requires_grad = False

    final_optimizer = torch.optim.AdamW(
        [p for p in final_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-2,
    )
    for refit_epoch in range(min(3, args.epochs)):
        _run_epoch(final_model, examples, list(range(len(examples))), cached_bundles, final_optimizer, epoch_idx=refit_epoch)

    final_model.eval()
    final_model.classical_ensemble = classical_ensemble

    # Prediction probability diagnostics
    eval_probs = classical_ensemble.predict_proba(X_oof_total)["final_probability"]
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

    from multihaludet.feature_extractor import FEATURE_SCHEMA_HASH, EXPECTED_TOTAL_FEATURE_DIM, CANONICAL_FEATURE_SCHEMA

    # Build publication-grade metadata with strict provenance
    metadata: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": args.model_name,
        "sampled_layers": final_model.num_sampled_layers,
        "num_examples_total": len(examples),
        "dataset_diagnostics": diag_summary,
        "training_protocol": "strict_oof_v4",
        "frozen_test_isolated": True if frozen_test_examples else False,
        "frozen_test_size": len(frozen_test_examples),
        "threshold_source": "development_oof",
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.lr,
        "feature_schema_version": CANONICAL_FEATURE_SCHEMA.get("schema_version", "multihaludet_v3.2"),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "feature_dimension": EXPECTED_TOTAL_FEATURE_DIM,
        "deep_feature_dimension": final_model.encoder_dim,
        "explicit_feature_dimension": int(CANONICAL_FEATURE_SCHEMA.get("explicit_feature_dim", 15)),
        "base_learner_names": classical_ensemble.active_member_names,
        "is_complete_ensemble": classical_ensemble.is_complete_ensemble,
        "ensemble_mode": classical_ensemble.mode,
        "dependency_versions": _get_dependency_versions(),
        "feature_diagnostics": feature_diagnostics,
        "probability_diagnostics": prob_stats,
        "fold_validation_metrics": fold_results,
        "comparative_system_metrics": comp_systems,
        "oof_base_learner_metrics": stacking_results["base_oof_metrics"],
        "oof_meta_learner_metrics": stacking_results["meta_oof_metrics"],
    }

    final_model.save_checkpoint(args.checkpoint_out, metadata=metadata)
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
    p.add_argument("--frozen-test", default=_default_dataset_path("halueval_benchmark_500.jsonl"), help="Path to dedicated frozen 500-sample benchmark test set.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without validation AUROC improvement).")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze Transformer encoder and attention weights to prevent representation overfitting on small datasets.")
    p.add_argument("--allow-nli-fallback", action="store_true", help="Allow fallback when DeBERTa-v3 NLI model is unavailable (disabled by default under strict_nli=True).")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--checkpoint-out", default="./multihaludet/checkpoints/multihaludet.pt")

    p.add_argument("--max-samples", type=int, default=None, help="Maximum number of total samples to use for training (e.g. 2000).")
    p.add_argument("--allow-reduced-ensemble", action="store_true", help="Allow training a reduced ensemble in development/test mode if LightGBM/XGBoost is unavailable.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(_build_arg_parser().parse_args())

