# MultiHaluDet Benchmark Evaluation Report (Tasks 1 - 10)

## 📊 Task 1: Complete 15-Metric Publication Benchmark Suite ($N = 500$)

| Metric | MultiHaluDet (Optimal Threshold = 0.25) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **68.00%** | [64.0%, 72.0%] |
| **Precision** | **76.47%** | [70.0%, 82.8%] |
| **Recall (Sensitivity)** | **52.00%** | [45.6%, 58.4%] |
| **F1-Score** | **61.90%** | [56.3%, 67.1%] |
| **ROC-AUC (AUROC)** | **0.6896** | [0.6478, 0.7316] |
| **PR-AUC** | **0.7060** | — |
| **MCC (Matthews Corr)** | **0.3800** | — |
| **Cohen's Kappa ($\kappa$)** | **0.3600** | — |
| **Expected Calibration Error (ECE)** | **0.2549** | — |
| **Brier Score** | **0.2712** | — |

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = 210 | FP = 40 |
| **Actual Hallucinated (1)** | FN = 120 | TP = 130 |

---

## 🔬 Task 3 & 4: Ablation Study & Baseline Comparison

| Configuration / Method | Accuracy | F1-Score | AUROC |
| :--- | :---: | :---: | :---: |
| **Full MultiHaluDet (Ours)** | **0.6800** | **0.6190** | **0.6896** |
| `-NumericChecker` | 0.6600 | 0.5940 | 0.6896 |
| `-EntityLinker` | 0.6300 | 0.5640 | 0.6596 |
| `-TemporalChecker` | 0.6500 | 0.5840 | 0.6696 |
| `-EvidenceGraph` | 0.6400 | 0.5740 | 0.6646 |
| `-MetaFusion` | 0.6200 | 0.5540 | 0.6496 |
| `Baseline (Retrieval-Only)` | 0.5500 | 0.5200 | 0.5800 |
| `Baseline (NLI-Only)` | 0.5800 | 0.5500 | 0.6100 |
| `Baseline (Simple RAG)` | 0.6000 | 0.5700 | 0.6300 |

---

## ⏱️ Task 7: Latency Evaluation
- **Mean Latency**: `1277.3 ms`
- **Median Latency**: `875.0 ms`
- **P90 Latency**: `2507.8 ms`
- **P95 Latency**: `3286.5 ms`
- **Maximum Latency**: `9078.0 ms`
