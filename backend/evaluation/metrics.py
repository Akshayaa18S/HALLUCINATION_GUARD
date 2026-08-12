"""
Evaluation Layer - Publication Metric Suite.

Computes classification, retrieval, calibration, and latency metrics:
- Accuracy, Precision, Recall, F1, AUROC
- Retrieval: Recall@K, MRR, nDCG
- Calibration: ECE (Expected Calibration Error), Brier Score
- Latency: Retrieval Latency, Inference Latency
"""

from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    """Computes Accuracy, Precision, Recall, F1, and AUROC."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    total = len(y_true)
    acc = (tp + tn) / float(total) if total > 0 else 0.0
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    try:
        from sklearn.metrics import average_precision_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.50
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.50
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0
        kappa = float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0
    except Exception:
        auroc = 0.50
        pr_auc = 0.50
        mcc = 0.0
        kappa = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def compute_retrieval_metrics(retrieved_ids: list[list[str]], ground_truth_ids: list[list[str]], k: int = 5) -> dict[str, float]:
    """Computes Recall@K, MRR (Mean Reciprocal Rank), and nDCG@K."""
    if not retrieved_ids or not ground_truth_ids:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    recalls = []
    mrrs = []
    ndcgs = []

    for r_list, gt_list in zip(retrieved_ids, ground_truth_ids):
        if not gt_list:
            continue
        gt_set = set(gt_list)
        top_k = r_list[:k]

        hits = sum(1 for doc in top_k if doc in gt_set)
        recalls.append(hits / float(len(gt_set)))

        mrr = 0.0
        for rank, doc in enumerate(top_k, 1):
            if doc in gt_set:
                mrr = 1.0 / float(rank)
                break
        mrrs.append(mrr)

        dcg = sum((1.0 / math.log2(rank + 1)) for rank, doc in enumerate(top_k, 1) if doc in gt_set)
        idcg = sum((1.0 / math.log2(rank + 1)) for rank in range(1, min(len(gt_set), k) + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
    }


def compute_calibration_metrics(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> dict[str, float]:
    """Computes Expected Calibration Error (ECE) and Brier Score."""
    if not y_true or not y_prob:
        return {"ece": 0.0, "brier_score": 0.0}

    # Brier Score
    brier = float(np.mean([(p - t) ** 2 for p, t in zip(y_prob, y_true)]))

    # ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = [(p, t) for p, t in zip(y_prob, y_true) if bin_lower <= p < bin_upper]
        if in_bin:
            bin_acc = np.mean([t for _, t in in_bin])
            bin_conf = np.mean([p for p, _ in in_bin])
            ece += (len(in_bin) / float(total)) * abs(bin_acc - bin_conf)

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier, 4),
    }
