"""
Unified Outer-Fold Multi-Seed OOF Comparative Validation Suite.
Evaluates Systems A, B, C, and D across seeds 42, 123, 456, and 789 on identical outer 5-fold CV splits.
Within each fold, validation samples are strictly excluded from neural training, feature scaling,
and classical ensemble fitting.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
import numpy as np
import torch
from typing import Any

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.training.datasets import load_halueval, load_triviaqa
from multihaludet.feature_extractor import ExplicitFeatureExtractor
from multihaludet.generation_backend import HFGenerationBackend
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.training.train import _run_epoch, _default_dataset_path, precompute_generation_bundles


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multiseed_oof")


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc as calc_auc
    y_pred = (y_prob >= 0.5).astype(int)
    has_two = len(set(y_true)) > 1
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(calc_auc(rec_arr, prec_arr)) if has_two else 0.5
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if has_two else 0.5,
        "pr_auc": pr_auc,
    }


def run_multiseed_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    logger.info("=== STARTING UNIFIED OUTER-FOLD MULTI-SEED OOF COMPARATIVE EXPERIMENT ===")
    logger.info("Seeds: %s | Folds: %d | Max Samples: %s | Resolved Device: %s", args.seeds, args.folds, args.max_samples, args.device)

    # 1. Collect development dataset examples
    examples = []
    if args.halueval_summarization and Path(args.halueval_summarization).exists():
        examples.extend(list(load_halueval(args.halueval_summarization, "summarization")))
    if args.halueval_qa and Path(args.halueval_qa).exists():
        examples.extend(list(load_halueval(args.halueval_qa, "qa")))
    if args.halueval_dialogue and Path(args.halueval_dialogue).exists():
        examples.extend(list(load_halueval(args.halueval_dialogue, "dialogue")))
    if args.triviaqa and Path(args.triviaqa).exists():
        examples.extend(list(load_triviaqa(args.triviaqa)))

    if args.max_samples and len(examples) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:args.max_samples]

    assert len(examples) > 0, "No development examples found for evaluation!"
    logger.info("Total development examples loaded: %d", len(examples))
    y_labels = np.array([1 if ex.label else 0 for ex in examples], dtype=np.int64)

    # 2. Precompute generation bundles for genuine neural feature extraction
    logger.info("Initializing generation backend: %s...", args.model_name)
    backend = HFGenerationBackend(model_name=args.model_name, device=args.device)

    model_slug = re.sub(r"[^\w\-]", "_", str(args.model_name))
    cache_path = Path(f"./multihaludet/data/bundle_cache_{model_slug}_{backend.hidden_size}.pt")
    if args.model_name != "fake" and cache_path.exists():
        logger.info("Loading cached generation bundles from %s...", cache_path)
        cached_bundles = torch.load(cache_path, weights_only=False)
    else:
        cached_bundles = precompute_generation_bundles(backend, examples)

    # 3. Extract 15 explicit non-trainable verification & NLI features
    strict_nli = not args.allow_nli_fallback
    extractor = ExplicitFeatureExtractor(device=args.device, strict_nli=strict_nli)
    logger.info("Extracting non-trainable explicit features across %d examples...", len(examples))
    explicit_feats = []
    for ex in examples:
        ev_texts = getattr(ex, "evidence_texts", None)
        vec = extractor.extract_feature_vector(ex.query, ex.response, evidence_texts=ev_texts)
        explicit_feats.append(vec)
    X_explicit_all = np.array(explicit_feats, dtype=np.float32)

    system_names = [
        "System_A_Qwen_Baseline",
        "System_B_DeBERTa_NLI_Only",
        "System_C_NLI_Plus_Evidence",
        "System_D_Full_Fused_MultiHaluDet",
    ]

    metrics_history: dict[str, dict[str, list[float]]] = {
        sys: {"auc": [], "pr_auc": [], "f1": [], "accuracy": [], "precision": [], "recall": []}
        for sys in system_names
    }
    seed_results: dict[int, dict[str, dict[str, float]]] = {}

    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # 4. Single outer-fold cross-validation loop per seed
    for seed in args.seeds:
        logger.info("\n--- Evaluating Seed %d (Unified Outer Fold OOF Protocol) ---", seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)

        oof_preds_dict = {sys_name: np.zeros(len(examples), dtype=np.float32) for sys_name in system_names}
        oof_written_dict = {sys_name: np.zeros(len(examples), dtype=bool) for sys_name in system_names}

        for fold_idx, (train_idx_arr, val_idx_arr) in enumerate(skf.split(examples, y_labels)):
            train_idx_list = train_idx_arr.tolist()
            val_idx_list = val_idx_arr.tolist()

            # Strict Outer-Fold Isolation Assertions
            assert set(train_idx_list).isdisjoint(set(val_idx_list)), f"Fold {fold_idx + 1} contamination error!"
            assert len(np.unique(val_idx_list)) == len(val_idx_list), f"Duplicate validation indices in fold {fold_idx + 1}"

            # Step A: Train neural MultiHaluDetModel strictly on outer train_idx
            fold_neural_model = MultiHaluDetModel(hidden_size=backend.hidden_size).to(args.device)
            fold_optimizer = torch.optim.AdamW(
                [p for p in fold_neural_model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-2
            )

            for epoch in range(min(3, args.epochs)):
                _run_epoch(fold_neural_model, examples, train_idx_list, cached_bundles, fold_optimizer, epoch_idx=epoch)

            # Step B: Extract deep features for train_idx and val_idx using the same fold neural model
            fold_neural_model.eval()
            with torch.no_grad():
                tr_deep_list, va_deep_list = [], []
                for idx in train_idx_list:
                    b = cached_bundles[idx]
                    fused_tr = fold_neural_model.compute_deep_features(b).cpu().numpy().reshape(-1) if not b.is_empty() else np.zeros(256, dtype=np.float32)
                    fnorm = float(np.linalg.norm(fused_tr, ord=2))
                    if fnorm > 1e-8:
                        fused_tr = fused_tr / fnorm
                    tr_deep_list.append(fused_tr)

                for idx in val_idx_list:
                    b = cached_bundles[idx]
                    fused_va = fold_neural_model.compute_deep_features(b).cpu().numpy().reshape(-1) if not b.is_empty() else np.zeros(256, dtype=np.float32)
                    fnorm = float(np.linalg.norm(fused_va, ord=2))
                    if fnorm > 1e-8:
                        fused_va = fused_va / fnorm
                    va_deep_list.append(fused_va)

            X_tr_deep = np.array(tr_deep_list, dtype=np.float32)
            X_va_deep = np.array(va_deep_list, dtype=np.float32)

            X_tr_explicit = X_explicit_all[train_idx_list]
            X_va_explicit = X_explicit_all[val_idx_list]

            y_tr = y_labels[train_idx_list]

            from multihaludet.feature_extractor import EXPLICIT_FEATURE_NAMES
            nli_indices = [EXPLICIT_FEATURE_NAMES.index(n) for n in ["nli_contradiction_score", "nli_entailment_score", "nli_neutral_score"]]

            # System-specific train/val feature slices for fold k
            system_slices = {
                "System_A_Qwen_Baseline": (X_tr_deep, X_va_deep, 256),
                "System_B_DeBERTa_NLI_Only": (X_tr_explicit[:, nli_indices], X_va_explicit[:, nli_indices], 3),
                "System_C_NLI_Plus_Evidence": (X_tr_explicit, X_va_explicit, 15),
                "System_D_Full_Fused_MultiHaluDet": (
                    np.concatenate([X_tr_deep, X_tr_explicit], axis=-1),
                    np.concatenate([X_va_deep, X_va_explicit], axis=-1),
                    271
                ),
            }

            # Step C: Fit classical ensemble strictly on outer train, predict outer val using inner-scaler
            for sys_name, (X_tr_sys, X_va_sys, exp_dim) in system_slices.items():
                ens = ClassicalEnsemble(
                    seed=seed,
                    allow_reduced_ensemble=True,
                    expected_feature_dim=exp_dim,
                    system_name=sys_name,
                )
                ens.fit_oof(X_tr_sys, y_tr, n_splits=args.folds, seed=seed)
                fold_probs = ens.predict_proba(X_va_sys)["final_probability"]

                assert not np.isnan(fold_probs).any(), f"NaN detected in predictions for {sys_name} on fold {fold_idx + 1}"

                oof_preds_dict[sys_name][val_idx_list] = fold_probs
                oof_written_dict[sys_name][val_idx_list] = True


        logger.info("Nested outer-fold OOF protocol completed for seed %d; validation samples were excluded from neural training, feature scaling, and classical fitting within each fold.", seed)


        seed_comp: dict[str, dict[str, float]] = {}
        for sys_name in system_names:
            assert np.all(oof_written_dict[sys_name]), f"OOF predictions incomplete for {sys_name}"
            assert np.isfinite(oof_preds_dict[sys_name]).all(), f"Non-finite values found in {sys_name}"
            assert len(np.unique(np.round(oof_preds_dict[sys_name], 4))) > 1, f"Degenerate predictions in {sys_name}"

            m = _compute_metrics(y_labels, oof_preds_dict[sys_name])
            seed_comp[sys_name] = m
            for metric_key in ["auc", "pr_auc", "f1", "accuracy", "precision", "recall"]:
                metrics_history[sys_name][metric_key].append(m[metric_key])

        seed_results[seed] = seed_comp

    # 5. Output Mean +/- Std Summary Table
    logger.info("\n=== MULTI-SEED OUT-OF-FOLD (OOF) SUMMARY TABLE (Seeds: %s) ===", args.seeds)
    header = f"{'System':<35} | {'AUROC Mean +/- Std':<20} | {'PR-AUC Mean +/- Std':<20} | {'F1 Mean +/- Std':<20} | {'Accuracy Mean +/- Std':<20}"
    logger.info(header)
    logger.info("-" * len(header))

    summary_output = {}
    for sys_name in system_names:
        sys_metrics = metrics_history[sys_name]
        auc_m, auc_s = np.mean(sys_metrics["auc"]), np.std(sys_metrics["auc"])
        pr_m, pr_s = np.mean(sys_metrics["pr_auc"]), np.std(sys_metrics["pr_auc"])
        f1_m, f1_s = np.mean(sys_metrics["f1"]), np.std(sys_metrics["f1"])
        acc_m, acc_s = np.mean(sys_metrics["accuracy"]), np.std(sys_metrics["accuracy"])

        summary_output[sys_name] = {
            "auc_mean": auc_m, "auc_std": auc_s,
            "pr_auc_mean": pr_m, "pr_auc_std": pr_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
        }

        row = (
            f"{sys_name:<35} | "
            f"{auc_m:.4f} +/- {auc_s:.4f}     | "
            f"{pr_m:.4f} +/- {pr_s:.4f}     | "
            f"{f1_m:.4f} +/- {f1_s:.4f}     | "
            f"{acc_m:.4f} +/- {acc_s:.4f}"
        )
        logger.info(row)

    return {"seed_results": seed_results, "summary": summary_output}


def _build_arg_parser() -> argparse.ArgumentParser:
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="Unified Outer-Fold Multi-Seed OOF Evaluator")
    p.add_argument("--halueval-qa", default=_default_dataset_path("halueval_qa.jsonl"))
    p.add_argument("--halueval-dialogue", default=_default_dataset_path("halueval_dialogue.jsonl"))
    p.add_argument("--halueval-summarization", default=_default_dataset_path("halueval_summarization.jsonl"))
    p.add_argument("--triviaqa", default=_default_dataset_path("triviaqa_labeled.jsonl"))
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default=default_dev)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--allow-nli-fallback", action="store_true")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    run_multiseed_evaluation(parser.parse_args())
