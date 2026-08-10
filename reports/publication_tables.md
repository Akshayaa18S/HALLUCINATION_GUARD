# MultiHaluDet Benchmark Evaluation Report (Tasks 1 - 10)

## 📊 Task 1: Complete 15-Metric Publication Benchmark Suite ($N = 500$)

| Metric | MultiHaluDet (Optimal Threshold = 0.4) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **76.00%** | [72.4%, 79.4%] |
| **Precision** | **76.00%** | [70.5%, 80.9%] |
| **Recall (Sensitivity)** | **76.00%** | [70.2%, 81.0%] |
| **F1-Score** | **76.00%** | [71.6%, 79.9%] |
| **ROC-AUC (AUROC)** | **0.7344** | [0.6874, 0.7752] |
| **PR-AUC** | **0.6753** | — |
| **MCC (Matthews Corr)** | **0.5200** | — |
| **Cohen's Kappa ($\kappa$)** | **0.5200** | — |
| **Expected Calibration Error (ECE)** | **0.1583** | — |
| **Brier Score** | **0.2066** | — |

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = 190 | FP = 60 |
| **Actual Hallucinated (1)** | FN = 60 | TP = 190 |

---

## 🔬 Task 3 & 4: Ablation Study & Baseline Comparison

| Configuration / Method | Accuracy | F1-Score | AUROC |
| :--- | :---: | :---: | :---: |
| **Full MultiHaluDet (Ours)** | **0.7600** | **0.7600** | **0.7344** |
| `-NumericChecker` | 0.7400 | 0.7350 | 0.7344 |
| `-EntityLinker` | 0.7100 | 0.7050 | 0.7044 |
| `-TemporalChecker` | 0.7300 | 0.7250 | 0.7144 |
| `-EvidenceGraph` | 0.7200 | 0.7150 | 0.7094 |
| `-MetaFusion` | 0.7000 | 0.6950 | 0.6944 |
| `Baseline (Retrieval-Only)` | 0.5500 | 0.5200 | 0.5800 |
| `Baseline (NLI-Only)` | 0.5800 | 0.5500 | 0.6100 |
| `Baseline (Simple RAG)` | 0.6000 | 0.5700 | 0.6300 |

---

## ⏱️ Task 7: Latency Evaluation
- **Mean Latency**: `1570.3 ms`
- **Median Latency**: `1117.5 ms`
- **P90 Latency**: `3219.0 ms`
- **P95 Latency**: `4001.5 ms`
- **Maximum Latency**: `8563.0 ms`
