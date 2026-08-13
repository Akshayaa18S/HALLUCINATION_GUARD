"""
System C V2 Development Experiment Runner (4 Seeds x 5 Outer Folds).

Executes nested fold-safe cross-validation strictly on development data across 4 seeds:
- Extract 22 fine-grained evidence/NLI features.
- Compute independent outer OOF predictions per seed.
- Calculate per-seed and overall summary (mean +- std, min, max) for Accuracy, F1, AUROC, PR-AUC, Precision, and Recall.
- Compute feature variance and correlation diagnostics across the 22 features.
- Evaluate simultaneous V2 development gate criteria.
"""

import sys
import logging
import random
from pathlib import Path
from typing import Any
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc as calc_auc,
)

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from multihaludet.feature_extractor_v2 import EXPLICIT_FEATURE_NAMES_V2, ExplicitFeatureExtractorV2
from multihaludet.ensemble_v2 import ClassicalEnsembleV2
from multihaludet.training.datasets import load_halueval, load_triviaqa
from retrieval.evidence_cache import get_evidence_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hallucination_guard.v2_runner")

SEEDS = [42, 123, 2024, 3407]


def load_development_dataset(max_samples: int = 2000) -> list[Any]:
    data_dir = backend_dir / "multihaludet" / "data"
    examples = []

    for filename, task in [("halueval_qa.jsonl", "qa"), ("halueval_summarization.jsonl", "summarization"), ("halueval_dialogue.jsonl", "dialogue")]:
        p = data_dir / filename
        if p.exists():
            examples.extend(list(load_halueval(str(p), task)))

    tqa_p = data_dir / "triviaqa_labeled.jsonl"
    if tqa_p.exists():
        examples.extend(list(load_triviaqa(str(tqa_p))))

    if max_samples and len(examples) > max_samples:
        random.seed(42)
        pos = [ex for ex in examples if ex.label]
        neg = [ex for ex in examples if not ex.label]
        random.shuffle(pos)
        random.shuffle(neg)
        half = max_samples // 2
        examples = pos[:half] + neg[:half]
        random.shuffle(examples)

    return examples


def compute_feature_diagnostics(X: np.ndarray) -> dict[str, Any]:
    variances = np.var(X, axis=0)
    low_var = [EXPLICIT_FEATURE_NAMES_V2[i] for i, v in enumerate(variances) if v <= 0.001]

    corr_matrix = np.corrcoef(X, rowvar=False)
    high_corr = []
    n = len(EXPLICIT_FEATURE_NAMES_V2)
    for i in range(n):
        for j in range(i + 1, n):
            val = float(corr_matrix[i, j])
            if abs(val) >= 0.85:
                high_corr.append((EXPLICIT_FEATURE_NAMES_V2[i], EXPLICIT_FEATURE_NAMES_V2[j], round(val, 4)))

    return {
        "low_variance_features": low_var,
        "highly_correlated_pairs": high_corr,
    }


def run_system_c_v2_development():
    logger.info("=== STARTING SYSTEM C V2 DEVELOPMENT EXPERIMENT (4 SEEDS x 5 OUTER FOLDS) ===")

    examples = load_development_dataset(max_samples=2000)
    logger.info("Loaded %d balanced development examples", len(examples))
    y_dev = np.array([1 if ex.label else 0 for ex in examples], dtype=np.int64)

    logger.info("Extracting 22 Fine-Grained Features using Evidence Cache...")
    extractor = ExplicitFeatureExtractorV2(strict_nli=False)
    evidence_cache = get_evidence_cache()

    explicit_feats = []
    for i, ex in enumerate(examples):
        ev_texts = getattr(ex, "evidence_texts", None)
        if not ev_texts:
            cached_snips = evidence_cache.get(ex.query)
            if cached_snips:
                ev_texts = [s.get("text", "") for s in cached_snips if s.get("text")]
        vec = extractor.extract_feature_vector_v2(ex.query, ex.response, evidence_texts=ev_texts)
        explicit_feats.append(vec)

    X_dev = np.array(explicit_feats, dtype=np.float32)
    logger.info("Extracted Feature Matrix X_dev shape: %s", X_dev.shape)

    diagnostics = compute_feature_diagnostics(X_dev)

    per_seed_results = []
    for seed in SEEDS:
        logger.info("Running Seed %d (5 Outer Folds)...", seed)
        ensemble = ClassicalEnsembleV2(seed=seed, expected_feature_dim=22)
        res = ensemble.fit_oof_nested(X_dev, y_dev, n_splits=5, seed=seed)

        oof_probs = np.array(res["oof_probabilities"], dtype=np.float32)
        tau = float(res["optimal_threshold"])
        oof_preds = (oof_probs >= tau).astype(int)

        acc = float(accuracy_score(y_dev, oof_preds))
        prec = float(precision_score(y_dev, oof_preds, zero_division=0))
        rec = float(recall_score(y_dev, oof_preds, zero_division=0))
        f1 = float(f1_score(y_dev, oof_preds, zero_division=0))
        auc = float(roc_auc_score(y_dev, oof_probs))

        prec_arr, rec_arr, _ = precision_recall_curve(y_dev, oof_probs)
        pr_auc = float(calc_auc(rec_arr, prec_arr))

        per_seed_results.append({
            "seed": seed,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "pr_auc": pr_auc,
            "threshold": tau,
        })

    # Summary Statistics across 4 Seeds
    accs = [r["accuracy"] * 100 for r in per_seed_results]
    f1s = [r["f1"] * 100 for r in per_seed_results]
    aucs = [r["auc"] * 100 for r in per_seed_results]
    pr_aucs = [r["pr_auc"] * 100 for r in per_seed_results]

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    mean_pr_auc, std_pr_auc = np.mean(pr_aucs), np.std(pr_aucs)

    # Multi-Metric Gate Evaluation (Simultaneous)
    gate_acc_pass = mean_acc >= 82.0
    gate_f1_pass = mean_f1 >= 80.0
    gate_auc_pass = mean_auc >= 82.0
    gate_std_pass = std_acc <= 0.50

    all_gates_pass = gate_acc_pass and gate_f1_pass and gate_auc_pass and gate_std_pass

    print("\n" + "=" * 70)
    print("SYSTEM C V2 DEVELOPMENT EVALUATION REPORT")
    print("=" * 70)
    print(f"Seeds        : {SEEDS}")
    print(f"Outer Folds  : 5 per seed (20 outer fold observations)")
    print(f"Feature Dim  : 22 Explicit Verification Features")
    print(f"Dev Samples  : {len(examples)}")

    print("\nPER-SEED RESULTS")
    print("-" * 70)
    print(f"{'Seed':<8} | {'Accuracy':<10} | {'F1-Score':<10} | {'AUROC':<10} | {'PR-AUC':<10} | {'Threshold':<10}")
    print("-" * 70)
    for r in per_seed_results:
        print(f"{r['seed']:<8} | {r['accuracy']*100:<9.2f}% | {r['f1']*100:<9.2f}% | {r['auc']*100:<9.2f}% | {r['pr_auc']*100:<9.2f}% | {r['threshold']:<10.4f}")

    print("\nSUMMARY (4-SEED MEAN ± STD)")
    print("-" * 70)
    print(f"Accuracy   : {mean_acc:.2f}% ± {std_acc:.2f}% (Min: {np.min(accs):.2f}%, Max: {np.max(accs):.2f}%)")
    print(f"F1-Score   : {mean_f1:.2f}% ± {std_f1:.2f}% (Min: {np.min(f1s):.2f}%, Max: {np.max(f1s):.2f}%)")
    print(f"AUROC      : {mean_auc:.2f}% ± {std_auc:.2f}% (Min: {np.min(aucs):.2f}%, Max: {np.max(aucs):.2f}%)")
    print(f"PR-AUC     : {mean_pr_auc:.2f}% ± {std_pr_auc:.2f}% (Min: {np.min(pr_aucs):.2f}%, Max: {np.max(pr_aucs):.2f}%)")

    print("\nFEATURE DIAGNOSTICS")
    print("-" * 70)
    print(f"Low-Variance Features (var <= 0.001) : {diagnostics['low_variance_features'] or 'None'}")
    print("Highly-Correlated Pairs (|r| >= 0.85) :")
    if diagnostics['highly_correlated_pairs']:
        for f1_name, f2_name, r_val in diagnostics['highly_correlated_pairs']:
            print(f"  - {f1_name} <--> {f2_name}: r = {r_val}")
    else:
        print("  None")

    print("\n" + "=" * 70)
    print("SYSTEM C V2 DEVELOPMENT GATE EVALUATION")
    print("=" * 70)
    print(f"1. Outer OOF Accuracy  >= 82.0% : {'PASS' if gate_acc_pass else 'FAIL'} ({mean_acc:.2f}%)")
    print(f"2. Outer OOF F1-Score  >= 80.0% : {'PASS' if gate_f1_pass else 'FAIL'} ({mean_f1:.2f}%)")
    print(f"3. Outer OOF AUROC     >= 82.0% : {'PASS' if gate_auc_pass else 'FAIL'} ({mean_auc:.2f}%)")
    print(f"4. Inter-Seed Std     <= 0.50% : {'PASS' if gate_std_pass else 'FAIL'} ({std_acc:.2f}%)")
    print("-" * 70)
    if all_gates_pass:
        print("FINAL DECISION : PASS (System C V2 meets all gates! Lock V2 & proceed to frozen 500)")
    elif mean_acc >= 79.0:
        print("FINAL DECISION : SUB-GATE IMPROVEMENT (Log findings; do NOT touch frozen 500)")
    else:
        print("FINAL DECISION : FAIL / NO IMPROVEMENT (Do NOT touch frozen 500)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_system_c_v2_development()
