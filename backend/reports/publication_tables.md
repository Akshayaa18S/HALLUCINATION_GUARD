# MultiHaluDet Benchmark Evaluation Report (v3.1 Frozen)

## 📊 Task 1: Frozen Test Benchmark Suite ($N = 500$, 4 Seeds Mean ± Std)

| Metric | MultiHaluDet (Mean ± Std) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **82.00% ± 0.00%** | [78.6%, 85.3%] |
| **Precision** | **83.33% ± 0.00%** | [78.9%, 87.9%] |
| **Recall (Sensitivity)** | **80.00% ± 0.00%** | [75.1%, 84.5%] |
| **F1-Score** | **81.63% ± 0.00%** | [78.0%, 85.2%] |
| **ROC-AUC (AUROC)** | **0.9360 ± 0.0000** | [0.9173, 0.9533] |
| **PR-AUC** | **0.9351 ± 0.0000** | — |
| **MCC (Matthews Corr)** | **0.6405 ± 0.0000** | — |
| **Cohen's Kappa ($\kappa$)** | **0.6400 ± 0.0000** | — |
| **Expected Calibration Error (ECE)** | **0.0980** | — |
| **Brier Score** | **0.0870** | — |

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | TN = 210 | FP = 40 |
| **Actual Hallucinated (1)** | FN = 50 | TP = 200 |

---

## ⏱️ Task 7: Latency Evaluation
- **Mean Latency**: `164.9 ms`
- **Median Latency**: `157.0 ms`
- **P90 Latency**: `187.0 ms`
- **P95 Latency**: `188.0 ms`
- **Maximum Latency**: `4594.0 ms`
