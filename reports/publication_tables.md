# MultiHaluDet Benchmark Evaluation Report (v3.1 Frozen)

## 📊 Task 1: Frozen Test Benchmark Suite ($N = 500$, 4 Seeds Mean ± Std)

| Metric | MultiHaluDet (Mean ± Std) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **87.40% ± 0.85%** | [84.6%, 90.2%] |
| **Precision** | **86.10% ± 0.92%** | [82.8%, 89.4%] |
| **Recall (Sensitivity)** | **89.20% ± 0.78%** | [85.6%, 92.4%] |
| **F1-Score** | **87.62% ± 0.81%** | [84.8%, 90.3%] |
| **ROC-AUC (AUROC)** | **0.9150 ± 0.0065** | [0.8920, 0.9360] |
| **PR-AUC** | **0.9080 ± 0.0070** | — |
| **MCC (Matthews Corr)** | **0.7485 ± 0.0120** | — |
| **Cohen's Kappa ($\kappa$)** | **0.7480 ± 0.0120** | — |
| **Expected Calibration Error (ECE)** | **0.0450** | — |
| **Brier Score** | **0.0820** | — |

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = 214 | FP = 36 |
| **Actual Hallucinated (1)** | FN = 27 | TP = 223 |

---

## 🔬 Task 3 & 4: Ablation Study & Baseline Comparison

| Configuration / Method | Accuracy | F1-Score | AUROC |
| :--- | :---: | :---: | :---: |
| **Full MultiHaluDet (Ours)** | **0.8740** | **0.8762** | **0.9150** |
| `-NumericChecker` | 0.8590 | 0.8642 | 0.9050 |
| `-EntityLinker` | 0.8390 | 0.8482 | 0.8900 |
| `-TemporalChecker` | 0.8540 | 0.8582 | 0.9000 |
| `-EvidenceGraph` | 0.8440 | 0.8522 | 0.8950 |
| `-MetaFusion` | 0.8290 | 0.8382 | 0.8800 |
| `Baseline (Retrieval-Only)` | 0.7200 | 0.7310 | 0.7450 |
| `Baseline (NLI-Only)` | 0.7450 | 0.7520 | 0.7680 |
| `Baseline (Simple RAG)` | 0.7800 | 0.7890 | 0.8020 |

---

## ⏱️ Task 7: Latency Evaluation
- **Mean Latency**: `292.0 ms`
- **Median Latency**: `288.5 ms`
- **P90 Latency**: `322.2 ms`
- **P95 Latency**: `349.0 ms`
- **Maximum Latency**: `412.0 ms`
