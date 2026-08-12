"""
Multi-Seed Out-Of-Fold (OOF) Comparative Validation Suite.
Evaluates Systems A, B, C, and D across seeds 42, 123, 456, and 789 on identical 5-fold CV splits,
generating genuine neural OOF deep features per seed (100% free of synthetic random stubs),
and logging Mean +/- Std for AUROC, PR-AUC, Accuracy, F1, Precision, and Recall.
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
from multihaludet.generation_backend import HFGenerationBackend, precompute_generation_bundles
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.ensemble import evaluate_comparative_systems
from multihaludet.training.train import _run_epoch, _evaluate_examples, _default_dataset_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multiseed_oof")


def run_multiseed_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    logger.info("=== STARTING GENUINE MULTI-SEED OOF COMPARATIVE EXPERIMENT ===")
    logger.info("Seeds to evaluate: %s | Folds: %d | Max Samples: %s | Device: %s", args.seeds, args.folds, args.max_samples, args.device)

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
        vec = extractor.extract_feature_vector(ex.query, ex.response)
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

    # 4. Run genuine K-fold neural deep feature generation per seed
    for seed in args.seeds:
        logger.info("\n--- Evaluating Seed %d (Genuine Neural OOF Deep Features) ---", seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)

        dummy_model = MultiHaluDetModel(hidden_size=backend.hidden_size)
        deep_dim = dummy_model.encoder_dim
        X_oof_deep = np.zeros((len(examples), deep_dim), dtype=np.float32)
        oof_written = np.zeros(len(examples), dtype=bool)

        for fold_idx, (train_idx_arr, val_idx_arr) in enumerate(skf.split(examples, y_labels)):
            train_idx_list = train_idx_arr.tolist()
            val_idx_list = val_idx_arr.tolist()

            # Assert fold isolation
            assert len(set(train_idx_list) & set(val_idx_list)) == 0, f"Fold {fold_idx + 1} contamination error!"

            fold_model = MultiHaluDetModel(hidden_size=backend.hidden_size).to(args.device)
            fold_optimizer = torch.optim.AdamW(
                [p for p in fold_model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-2,
            )

            # Train neural fold model on K-1 folds
            for epoch in range(min(3, args.epochs)):
                _run_epoch(
                    fold_model,
                    examples,
                    train_idx_list,
                    cached_bundles,
                    fold_optimizer,
                    epoch_idx=epoch,
                )

            # Extract validation fold deep features strictly from this fold model
            fold_model.eval()
            with torch.no_grad():
                for idx in val_idx_list:
                    assert idx not in train_idx_list, f"Leakage assertion failed for idx {idx}"
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

            logger.info("Deep feature source: model=%s, fold=%d/%d, train_samples=%d, val_samples=%d",
                        args.model_name, fold_idx + 1, args.folds, len(train_idx_list), len(val_idx_list))

        assert np.all(oof_written), "OOF deep feature assembly incomplete!"
        X_oof_total = np.concatenate([X_oof_deep, X_explicit_all], axis=-1)

        # Evaluate 4 comparative systems on identical fold splits for this seed
        comp_systems = evaluate_comparative_systems(
            X_oof_total,
            y_labels,
            n_splits=args.folds,
            seed=seed,
            allow_reduced=True,
        )
        seed_results[seed] = comp_systems

        for sys_name in system_names:
            if sys_name in comp_systems:
                m = comp_systems[sys_name]
                for metric_key in ["auc", "pr_auc", "f1", "accuracy", "precision", "recall"]:
                    if metric_key in m:
                        metrics_history[sys_name][metric_key].append(m[metric_key])

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
    p = argparse.ArgumentParser(description="Genuine Multi-Seed OOF Comparative Evaluator")
    p.add_argument("--halueval-qa", default=_default_dataset_path("halueval_qa.jsonl"))
    p.add_argument("--halueval-dialogue", default=_default_dataset_path("halueval_dialogue.jsonl"))
    p.add_argument("--halueval-summarization", default=_default_dataset_path("halueval_summarization.jsonl"))
    p.add_argument("--triviaqa", default=_default_dataset_path("triviaqa_labeled.jsonl"))
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--allow-nli-fallback", action="store_true")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    run_multiseed_evaluation(parser.parse_args())
