"""
Targeted Diagnostic Tool for Qwen Deep Feature Branch & System A / System D Analysis.
Runs comprehensive diagnostics on:
1. Deep feature variance, norms, and sparsity
2. System A fold model loss progression and OOF prediction distribution
3. Feature importance allocation in System D (256 deep vs 15 explicit)
4. Empirical explanation of why System D underperforms System C
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.training.datasets import load_halueval, load_triviaqa
from multihaludet.feature_extractor import ExplicitFeatureExtractor
from multihaludet.generation_backend import HFGenerationBackend
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.ensemble import ClassicalEnsemble
from multihaludet.training.train import _run_epoch, _default_dataset_path


def run_diagnostics():
    parser = argparse.ArgumentParser(description="Qwen Deep Branch Diagnostics")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    print("=== RUNNING TARGETED QWEN DEEP FEATURE DIAGNOSTICS ===")

    # 1. Load Dataset
    examples = []
    data_dir = backend_dir / "multihaludet" / "data"
    for filename, task in [("halueval_qa.jsonl", "qa"), ("halueval_summarization.jsonl", "summarization"), ("halueval_dialogue.jsonl", "dialogue")]:
        p = data_dir / filename
        if p.exists():
            examples.extend(list(load_halueval(str(p), task)))
    if len(examples) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(examples)
        examples = examples[:args.max_samples]

    y_labels = np.array([1 if ex.label else 0 for ex in examples], dtype=np.int64)
    print(f"Loaded {len(examples)} examples (Positive: {sum(y_labels)}, Negative: {len(y_labels) - sum(y_labels)})")

    # 2. Initialize Backend & Generate/Load Bundles
    data_dir = backend_dir / "multihaludet" / "data"
    cache_files = list(data_dir.glob("bundle_cache*.pt"))
    hidden_size = 2048
    if cache_files and args.model_name != "fake":
        c_path = cache_files[0]
        print(f"Loading cached Qwen bundles from {c_path.name}...")
        cached_bundles = torch.load(c_path, weights_only=False)
        if len(cached_bundles) > len(examples):
            cached_bundles = {i: cached_bundles[i] for i in range(len(examples))}
        if 0 in cached_bundles:
            hidden_size = cached_bundles[0].hidden_size
    else:
        backend = HFGenerationBackend(model_name=args.model_name, device=args.device)
        hidden_size = backend.hidden_size
        print(f"Generation Backend: {args.model_name} (Hidden Size: {hidden_size})")
        cached_bundles = {}
        for idx, ex in enumerate(examples):
            cached_bundles[idx] = backend.score_existing_response(ex.query, ex.response)

    # 3. Extract Deep Features Before Training
    neural_model = MultiHaluDetModel(hidden_size=hidden_size).to(args.device)
    neural_model.eval()

    deep_feats_untrained = []
    with torch.no_grad():
        for idx in range(len(examples)):
            b = cached_bundles[idx]
            if not b.is_empty():
                f = neural_model.compute_deep_features(b).cpu().numpy().reshape(-1)
            else:
                f = np.zeros(256, dtype=np.float32)
            deep_feats_untrained.append(f)

    X_deep_untrained = np.array(deep_feats_untrained, dtype=np.float32)

    print("\n--- 1. UNTRAINED DEEP FEATURE MATRIX DIAGNOSTICS ---")
    print(f"Matrix Shape: {X_deep_untrained.shape}")
    print(f"Mean: {np.mean(X_deep_untrained):.6f} | Std: {np.std(X_deep_untrained):.6f}")
    print(f"Min: {np.min(X_deep_untrained):.6f} | Max: {np.max(X_deep_untrained):.6f}")
    per_feat_std = np.std(X_deep_untrained, axis=0)
    print(f"Per-feature Std Dev (Mean across 256 dims): {np.mean(per_feat_std):.6f}")
    print(f"Zero-valued features percentage: {np.mean(X_deep_untrained == 0.0) * 100:.2f}%")

    # 4. Train Neural Model & Check Gradient/Loss Progression
    print("\n--- 2. NEURAL TRAINING PROGRESSION (3 EPOCHS) ---")
    optimizer = torch.optim.AdamW(neural_model.parameters(), lr=1e-4, weight_decay=1e-2)
    indices = list(range(len(examples)))
    for epoch in range(3):
        loss = _run_epoch(neural_model, examples, indices, cached_bundles, optimizer, epoch_idx=epoch)
        print(f"Epoch {epoch + 1}: BCE Loss = {loss:.6f}")

    # Extract Deep Features After Training
    neural_model.eval()
    deep_feats_trained = []
    with torch.no_grad():
        for idx in range(len(examples)):
            b = cached_bundles[idx]
            if not b.is_empty():
                f = neural_model.compute_deep_features(b).cpu().numpy().reshape(-1)
                fnorm = float(np.linalg.norm(f, ord=2))
                if fnorm > 1e-8:
                    f = f / fnorm
            else:
                f = np.zeros(256, dtype=np.float32)
            deep_feats_trained.append(f)
    X_deep_trained = np.array(deep_feats_trained, dtype=np.float32)

    print("\n--- 3. TRAINED DEEP FEATURE MATRIX DIAGNOSTICS ---")
    print(f"Normalized Matrix Mean: {np.mean(X_deep_trained):.6f} | Std: {np.std(X_deep_trained):.6f}")
    print(f"Per-feature Std Dev (Mean across 256 dims): {np.mean(np.std(X_deep_trained, axis=0)):.6f}")

    # 5. Extract 15 Explicit Features
    extractor = ExplicitFeatureExtractor(strict_nli=False)
    explicit_feats = []
    for ex in examples:
        ev_texts = getattr(ex, "evidence_texts", None)
        vec = extractor.extract_feature_vector(ex.query, ex.response, evidence_texts=ev_texts)
        explicit_feats.append(vec)
    X_explicit = np.array(explicit_feats, dtype=np.float32)

    print("\n--- 4. EXPLICIT FEATURE MATRIX DIAGNOSTICS ---")
    print(f"Explicit Matrix Shape: {X_explicit.shape}")
    print(f"Mean: {np.mean(X_explicit):.6f} | Std: {np.std(X_explicit):.6f}")

    # 6. Fit Classical Ensembles for System A, System C, and System D
    print("\n--- 5. SYSTEM CLASSIFIER COMPARISON ---")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    # System A (Qwen deep only)
    rf_A = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_A.fit(X_deep_trained, y_labels)
    prob_A = rf_A.predict_proba(X_deep_trained)[:, 1]
    auc_A = roc_auc_score(y_labels, prob_A)
    print(f"System A (Qwen Deep [256]) In-Sample AUROC: {auc_A:.4f} | Prob Std: {np.std(prob_A):.4f}")

    # System C (Explicit 15 only)
    rf_C = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_C.fit(X_explicit, y_labels)
    prob_C = rf_C.predict_proba(X_explicit)[:, 1]
    auc_C = roc_auc_score(y_labels, prob_C)
    print(f"System C (Explicit [15]) In-Sample AUROC: {auc_C:.4f} | Prob Std: {np.std(prob_C):.4f}")

    # System D (Fused 271)
    X_fused = np.concatenate([X_deep_trained, X_explicit], axis=-1)
    rf_D = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_D.fit(X_fused, y_labels)
    prob_D = rf_D.predict_proba(X_fused)[:, 1]
    auc_D = roc_auc_score(y_labels, prob_D)
    print(f"System D (Fused [271]) In-Sample AUROC: {auc_D:.4f} | Prob Std: {np.std(prob_D):.4f}")

    # Analyze feature importances in System D
    importances = rf_D.feature_importances_
    deep_imp_sum = float(np.sum(importances[:256]))
    explicit_imp_sum = float(np.sum(importances[256:]))
    print("\n--- 6. SYSTEM D FEATURE IMPORTANCE DILUTION ANALYSIS ---")
    print(f"Sum of Importances for 256 Deep Features: {deep_imp_sum:.4f} ({deep_imp_sum * 100:.2f}%)")
    print(f"Sum of Importances for 15 Explicit Features: {explicit_imp_sum:.4f} ({explicit_imp_sum * 100:.2f}%)")
    print(f"Average Importance per Deep Feature: {deep_imp_sum / 256:.6f}")
    print(f"Average Importance per Explicit Feature: {explicit_imp_sum / 15:.6f}")

    top_10_idx = np.argsort(-importances)[:10]
    print("\nTop 10 Most Important Features in System D:")
    for rank, idx in enumerate(top_10_idx, 1):
        if idx >= 256:
            fname = f"EXPLICIT_{idx - 256}"
        else:
            fname = f"DEEP_dim_{idx}"
        print(f"  Rank {rank}: {fname} (Importance: {importances[idx]:.4f})")

    print("\n=== DIAGNOSTICS COMPLETE ===")


if __name__ == "__main__":
    run_diagnostics()
