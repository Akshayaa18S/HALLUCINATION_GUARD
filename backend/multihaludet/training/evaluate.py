"""
Evaluates a trained MultiHaluDet checkpoint (feature extractor + classical ensemble)
on HaluEval, TriviaQA, and/or multilingual splits.

Generates evaluation metrics, empirical threshold tuning diagnostics, fusion weight sweeps,
and saves publication-ready plots (confusion matrix, ROC curve, precision-recall curve).

Usage:
    python -m multihaludet.training.evaluate \
        --checkpoint ./multihaludet/checkpoints/multihaludet.pt \
        --halueval-qa ./multihaludet/data/halueval_qa.jsonl \
        --generate-plots
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from multihaludet.generation_backend import HFGenerationBackend
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.training.datasets import HallucinationExample, load_halueval, load_multilingual, load_triviaqa

logger = logging.getLogger("hallucination_guard.multihaludet.evaluate")


def generate_publication_plots(
    y_true: list[int],
    y_prob: list[float],
    best_threshold: float = 0.5,
    output_dir: str = "./logs/plots",
) -> None:
    """Generates and saves publication-ready plots:
    1. confusion_matrix.png
    2. roc_curve.png
    3. precision_recall_curve.png
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed. Skipping plot generation.")
        return

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= best_threshold).astype(int)

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Factual", "Hallucinated"],
        yticklabels=["Factual", "Hallucinated"],
        title=f"MultiHaluDet Confusion Matrix (Threshold = {best_threshold:.2f})",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )
    fig.tight_layout()
    cm_path = out_p / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Confusion Matrix plot to %s", cm_path)

    # 2. ROC Curve Plot
    if len(set(y_true_arr)) > 1:
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        auc_val = roc_auc_score(y_true_arr, y_prob_arr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"MultiHaluDet (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        roc_path = out_p / "roc_curve.png"
        fig.savefig(roc_path, dpi=300)
        plt.close(fig)
        logger.info("Saved ROC Curve plot to %s", roc_path)

        # 3. Precision-Recall Curve Plot
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall_vals, precision_vals, color="blue", lw=2, label="MultiHaluDet PR Curve")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        pr_path = out_p / "precision_recall_curve.png"
        fig.savefig(pr_path, dpi=300)
        plt.close(fig)
        logger.info("Saved Precision-Recall Curve plot to %s", pr_path)


def benchmark_fusion_weights(
    y_true: list[int],
    y_prob_internal: list[float],
    y_prob_external: list[float] | None = None,
    weights: list[float] | None = None,
) -> None:
    """Benchmarks multiple fusion weights between internal model signal and external RAG signal."""
    w_list = weights or [0.50, 0.60, 0.70, 0.80, 0.90]
    y_true_arr = np.array(y_true, dtype=int)
    y_int_arr = np.array(y_prob_internal, dtype=float)

    # Actual external RAG scores required for publication benchmarks
    if not y_prob_external or len(y_prob_external) != len(y_true):
        logger.warning(
            "SKIPPING FUSION WEIGHT BENCHMARK: No real external RAG evidence scores provided. "
            "Synthetic ground-truth noise generation (y_true + noise) is strictly disabled in publication evaluations."
        )
        return

    y_ext_arr = np.array(y_prob_external, dtype=float)

    logger.info("=" * 80)
    logger.info("FUSION WEIGHT BENCHMARKING TABLE (Paper Insertion)")
    logger.info("=" * 80)
    logger.info(f"{'Internal Weight':<18} | {'External Weight':<18} | {'Accuracy':<10} | {'F1 Score':<10} | {'AUC':<10}")
    logger.info("-" * 80)

    for w in w_list:
        w_ext = round(1.0 - w, 2)
        fused_prob = w * y_int_arr + w_ext * y_ext_arr
        preds = (fused_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true_arr, preds)
        f1 = f1_score(y_true_arr, preds, zero_division=0)
        auc = roc_auc_score(y_true_arr, fused_prob) if len(set(y_true_arr)) > 1 else 0.5

        logger.info(f"{w:<18.2f} | {w_ext:<18.2f} | {acc:<10.4f} | {f1:<10.4f} | {auc:<10.4f}")
        logger.debug(f"Weight={w:.2f} | Internal={y_int_arr.mean():.4f} | External={y_ext_arr.mean():.4f} | Fused={fused_prob.mean():.4f}")
    logger.info("=" * 80)



def evaluate_split(
    model: MultiHaluDetModel, backend: HFGenerationBackend, examples: list[HallucinationExample]
) -> dict[str, Any]:
    total = len(examples)
    logger.info("Evaluating split containing %d examples...", total)

    y_true: list[int] = []
    y_prob: list[float] = []
    member_probs_dict: dict[str, list[float]] = {}

    for i, ex in enumerate(examples):
        bundle = backend.score_existing_response(ex.query, ex.response)
        if bundle.is_empty():
            logger.warning("Skipping empty-response example (source=%s)", ex.source)
            continue

        result = model.forward(bundle)
        y_true.append(1 if ex.label else 0)
        y_prob.append(float(result["internal_hallucination_probability"]))

        for name, prob in result["ensemble_member_probabilities"].items():
            if name not in member_probs_dict:
                member_probs_dict[name] = []
            member_probs_dict[name].append(float(prob))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info("Evaluation progress: %d/%d examples processed", i + 1, total)

    if not y_true:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "individual_base_learner_aucs": {},
            "n": 0,
            "y_true": [],
            "y_prob": [],
            "member_probs_dict": {},
        }

    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= model.decision_threshold).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
    has_two_classes = len(set(y_true_arr)) > 1

    base_aucs: dict[str, float] = {}
    for name, probs in member_probs_dict.items():
        if has_two_classes and len(probs) == len(y_true_arr):
            base_aucs[name] = float(roc_auc_score(y_true_arr, probs))
        else:
            base_aucs[name] = 0.5

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "auc": float(roc_auc_score(y_true_arr, y_prob_arr)) if has_two_classes else 0.5,
        "confusion_matrix": cm,
        "individual_base_learner_aucs": base_aucs,
        "n": len(y_true),
        "y_true": y_true,
        "y_prob": y_prob,
        "member_probs_dict": member_probs_dict,
    }


def run_threshold_tuning(
    y_true: list[int],
    y_prob: list[float],
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Sweeps probability decision thresholds to find the optimal decision cutoff for F1-score."""
    if not y_true or len(set(y_true)) <= 1:
        return {"best_threshold": 0.5, "best_f1": 0.0, "table": []}

    t_list = thresholds or [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70
    ]

    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    logger.info("=" * 75)
    logger.info("EMPIRICAL THRESHOLD TUNING DIAGNOSTICS (ROC-AUC = %.4f)", roc_auc)
    logger.info("=" * 75)

    best_threshold = 0.5
    best_f1 = -1.0
    table = []

    for t in t_list:
        preds = (y_prob_arr >= t).astype(int)
        acc = float(accuracy_score(y_true_arr, preds))
        prec = float(precision_score(y_true_arr, preds, zero_division=0))
        rec = float(recall_score(y_true_arr, preds, zero_division=0))
        f1 = float(f1_score(y_true_arr, preds, zero_division=0))

        logger.info(
            "Threshold %.2f | Acc %.4f | Prec %.4f | Recall %.4f | F1 %.4f",
            t, acc, prec, rec, f1
        )
        table.append({"threshold": t, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    logger.info("-" * 75)
    logger.info("EMPIRICALLY TUNED BEST THRESHOLD = %.2f | BEST F1 = %.4f", best_threshold, best_f1)
    logger.info("=" * 75)

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "roc_auc": roc_auc,
        "table": table,
    }


def compute_bootstrap_ci(
    y_true: list[int], y_prob: list[float], n_bootstraps: int = 1000, seed: int = 42
) -> dict[str, tuple[float, float]]:
    """Calculates 95% bootstrap confidence intervals for Accuracy, F1, and ROC-AUC."""
    rng = np.random.RandomState(seed)
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_prob, dtype=float)
    n = len(y_t)
    accs, f1s, aucs = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(y_t[idx])) <= 1:
            continue
        preds = (y_p[idx] >= 0.5).astype(int)
        accs.append(accuracy_score(y_t[idx], preds))
        f1s.append(f1_score(y_t[idx], preds, zero_division=0))
        aucs.append(roc_auc_score(y_t[idx], y_p[idx]))

    return {
        "accuracy_95ci": (round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)) if accs else (0.0, 0.0),
        "f1_95ci": (round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)) if f1s else (0.0, 0.0),
        "auc_95ci": (round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)) if aucs else (0.0, 0.0),
    }


def main(args: argparse.Namespace) -> None:
    from multihaludet.training.datasets import load_frozen_benchmark

    backend = HFGenerationBackend(model_name=args.model_name, device=args.device)
    model = MultiHaluDetModel(hidden_size=backend.hidden_size)
    if not model.load_checkpoint(args.checkpoint):
        raise SystemExit(f"No checkpoint found at {args.checkpoint} - train one first via training/train.py")
    model.eval()

    # Hard Provenance Guard Verification
    meta = getattr(model, "metadata", {}) or {}
    if meta and meta.get("threshold_source") and meta.get("threshold_source") != "development_oof":
        raise ValueError(
            "EVALUATION GUARD ERROR: Checkpoint decision threshold was not derived strictly from development OOF predictions! "
            "Test-set threshold tuning or invalid threshold sources are strictly forbidden in publication runs."
        )

    splits: dict[str, list[HallucinationExample]] = {}
    if args.frozen_test and Path(args.frozen_test).exists():
        splits["frozen_500_benchmark"] = load_frozen_benchmark(args.frozen_test)
    elif args.halueval_qa:
        splits["halueval_qa"] = list(load_halueval(args.halueval_qa, task="qa"))
    if args.halueval_dialogue and "frozen_500_benchmark" not in splits:
        splits["halueval_dialogue"] = list(load_halueval(args.halueval_dialogue, task="dialogue"))
    if args.halueval_summarization and "frozen_500_benchmark" not in splits:
        splits["halueval_summarization"] = list(load_halueval(args.halueval_summarization, task="summarization"))
    if args.triviaqa and "frozen_500_benchmark" not in splits:
        splits["triviaqa"] = list(load_triviaqa(args.triviaqa))
    if args.french:
        splits["french"] = list(load_multilingual(args.french, "fr"))
    if args.bangla:
        splits["bangla"] = list(load_multilingual(args.bangla, "bn"))
    if args.amharic:
        splits["amharic"] = list(load_multilingual(args.amharic, "am"))

    max_s = getattr(args, "max_samples", None)
    all_y_true: list[int] = []
    all_y_prob: list[float] = []

    for name, examples in splits.items():
        if max_s and max_s > 0:
            examples = examples[:max_s]
        metrics = evaluate_split(model, backend, examples)
        all_y_true.extend(metrics.get("y_true", []))
        all_y_prob.extend(metrics.get("y_prob", []))

        ci_dict = compute_bootstrap_ci(metrics.get("y_true", []), metrics.get("y_prob", []))

        logger.info("=== EVALUATION METRICS FOR SPLIT [%s] ===", name)
        logger.info("  Total Evaluated: %d", metrics["n"])
        logger.info("  Accuracy: %.4f (95%% CI: %s)", metrics["accuracy"], ci_dict["accuracy_95ci"])
        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall: %.4f", metrics["recall"])
        logger.info("  F1: %.4f (95%% CI: %s)", metrics["f1"], ci_dict["f1_95ci"])
        logger.info("  ROC-AUC: %.4f (95%% CI: %s)", metrics["auc"], ci_dict["auc_95ci"])
        logger.info("  Confusion Matrix: %s", metrics["confusion_matrix"])
        logger.info("  Base Learner AUCs: %s", metrics["individual_base_learner_aucs"])

        if metrics.get("y_true") and metrics.get("y_prob"):
            run_threshold_tuning(metrics["y_true"], metrics["y_prob"])

    if all_y_true:
        y_true_arr = np.array(all_y_true, dtype=int)
        y_prob_arr = np.array(all_y_prob, dtype=float)

        tuning_res = run_threshold_tuning(all_y_true, all_y_prob)
        # Use saved threshold from model, do NOT overwrite with test-tuned best_t
        eval_t = getattr(model, "decision_threshold", 0.5)

        benchmark_fusion_weights(all_y_true, all_y_prob)

        if args.generate_plots:
            logger.info("Generating publication plots in %s...", args.plots_dir)
            generate_publication_plots(all_y_true, all_y_prob, best_threshold=eval_t, output_dir=args.plots_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frozen-test", default=_default_dataset_path("halueval_benchmark_500.jsonl"), help="Path to dedicated frozen 500-sample benchmark test set.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--halueval-qa", default=_default_dataset_path("halueval_qa.jsonl"))
    p.add_argument("--halueval-dialogue", default=_default_dataset_path("halueval_dialogue.jsonl"))
    p.add_argument("--halueval-summarization", default=_default_dataset_path("halueval_summarization.jsonl"))
    p.add_argument("--triviaqa", default=_default_dataset_path("triviaqa_labeled.jsonl"))
    p.add_argument("--french")
    p.add_argument("--bangla")
    p.add_argument("--amharic")
    p.add_argument("--max-samples", type=int, default=None, help="Maximum examples per split to evaluate")
    p.add_argument("--generate-plots", action="store_true", help="Automatically save confusion matrix, ROC curve & PR curve figures")
    p.add_argument("--plots-dir", default="./logs/plots", help="Directory to save generated publication plot figures")
    return p



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(_build_arg_parser().parse_args())
