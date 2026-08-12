"""
Unified Baseline Evaluator for Publication Pipeline.
Evaluates all 9 baselines on Validation and Frozen Test splits.
Calculates AUROC, AUPRC, F1, Precision, and Recall.
Exports formatted Markdown and LaTeX publication tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from multihaludet.baselines import BaselineRegistry
from multihaludet.baselines.majority_random import MajorityClassBaseline, UniformRandomBaseline
from multihaludet.baselines.feature_probes import FeatureProbeLogReg, FeatureProbeXGBoost, SimpleHiddenProbe
from multihaludet.baselines.selfcheckgpt_probe import SelfCheckGPTBaseline
from multihaludet.baselines.semantic_entropy import SemanticEntropyBaseline
from multihaludet.baselines.retrieval_baselines import RetrievalOnlyBaseline, NLIOnlyBaseline, RetrievalPlusNLIBaseline

logger = logging.getLogger("hallucination_guard.baselines.evaluator")


@dataclass
class BaselineEvaluationResult:
    baseline_name: str
    roc_auc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    decision_threshold: float = 0.50


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    baseline_name: str,
    threshold: float = 0.50,
) -> BaselineEvaluationResult:
    """Computes publication metrics including AUPRC."""
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)

    # Fallback if single class present
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.50

    try:
        auprc = float(average_precision_score(y_true, y_probs))
    except Exception:
        auprc = float(np.mean(y_true))

    y_preds = (y_probs >= threshold).astype(int)

    f1 = float(f1_score(y_true, y_preds, zero_division=0))
    prec = float(precision_score(y_true, y_preds, zero_division=0))
    rec = float(recall_score(y_true, y_preds, zero_division=0))

    return BaselineEvaluationResult(
        baseline_name=baseline_name,
        roc_auc=roc_auc,
        auprc=auprc,
        f1=f1,
        precision=prec,
        recall=rec,
        decision_threshold=threshold,
    )


def register_all_baselines() -> List[Any]:
    """Registers all 9 baselines into the BaselineRegistry."""
    baselines = [
        MajorityClassBaseline(),
        UniformRandomBaseline(seed=42),
        FeatureProbeLogReg(seed=42),
        FeatureProbeXGBoost(seed=42),
        SimpleHiddenProbe(seed=42),
        SelfCheckGPTBaseline(seed=42),
        SemanticEntropyBaseline(),
        RetrievalOnlyBaseline(),
        NLIOnlyBaseline(),
        RetrievalPlusNLIBaseline(),
    ]
    for b in baselines:
        BaselineRegistry.register(b)
    return baselines


def run_baseline_evaluation(
    train_queries: List[str],
    train_responses: List[str],
    train_labels: List[int],
    eval_queries: List[str],
    eval_responses: List[str],
    eval_labels: List[int],
    train_features: np.ndarray | None = None,
    eval_features: np.ndarray | None = None,
) -> List[BaselineEvaluationResult]:
    """Fits and evaluates all registered baselines."""
    baselines = register_all_baselines()
    results: List[BaselineEvaluationResult] = []

    for b in baselines:
        logger.info("Evaluating baseline: %s", b.name)
        b.fit(train_queries, train_responses, train_labels, features=train_features)
        probs = b.predict_proba(eval_queries, eval_responses, features=eval_features)
        res = compute_baseline_metrics(eval_labels, probs, baseline_name=b.name)
        results.append(res)
        logger.info("  %s -> ROC-AUC: %.4f | AUPRC: %.4f | F1: %.4f", b.name, res.roc_auc, res.auprc, res.f1)

    return results


def export_baseline_tables(results: List[BaselineEvaluationResult], output_dir: str | Path = "./reports") -> None:
    """Exports baseline results to Markdown and LaTeX publication tables."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Markdown Table
    md_lines = [
        "# Publication Baseline Comparison Table",
        "",
        "| Baseline Method | ROC-AUC | AUPRC | F1 Score | Precision | Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        md_lines.append(f"| {r.baseline_name} | {r.roc_auc:.4f} | {r.auprc:.4f} | {r.f1:.4f} | {r.precision:.4f} | {r.recall:.4f} |")

    (out_path / "baseline_comparison_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    # LaTeX Table
    tex_lines = [
        "% Table 2: Main Baseline Comparison",
        "\\begin{table}[htbp]",
        "\\caption{Main Baseline Performance Comparison on Test Set}",
        "\\begin{center}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{ROC-AUC} & \\textbf{AUPRC} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} \\\\",
        "\\midrule",
    ]
    for r in results:
        tex_lines.append(f"{r.baseline_name.replace('_', ' ')} & {r.roc_auc:.4f} & {r.auprc:.4f} & {r.f1:.4f} & {r.precision:.4f} & {r.recall:.4f} \\\\")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:baselines}",
        "\\end{center}",
        "\\end{table}",
    ])

    (out_path / "baseline_comparison_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    logger.info("Saved baseline comparison tables in %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Synthetic smoke test evaluation
    tr_q = ["Q1", "Q2", "Q3", "Q4"]
    tr_r = ["R1", "R2", "R3", "R4"]
    tr_y = [1, 0, 1, 0]

    te_q = ["Q5", "Q6", "Q7", "Q8"]
    te_r = ["R5", "R6", "R7", "R8"]
    te_y = [1, 0, 1, 0]

    tr_feat = np.random.randn(4, 128)
    te_feat = np.random.randn(4, 128)

    res = run_baseline_evaluation(tr_q, tr_r, tr_y, te_q, te_r, te_y, tr_feat, te_feat)
    export_baseline_tables(res)
