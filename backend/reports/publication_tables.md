# MultiHaluDet Benchmark Evaluation Report (v3.1 Publication Ready)

> [!NOTE]
> **Methodology Note on Experimental Setup**:
> High detection performance ($\text{Accuracy} = 87.40\%$, $\text{AUROC} = 0.9150$) is achieved by training the 5-member classical stacking ensemble (`RandomForest`, `XGBoost`, `LightGBM`, `LogisticRegression`, `SVM`) on canonical normalized $D=265$ deep feature vectors (combining $D=200$ multi-scale transformer hidden states with $D=65$ explicit grounded verification features) using 5-Fold Stratified Out-Of-Fold (OOF) cross-validation and Youden's J statistic threshold optimization ($\tau^* = 0.10$).

---

## 📊 Table 1: Primary Frozen Test Benchmark Suite ($N = 500$, 4 Seeds Mean ± Std)

| Metric | MultiHaluDet (Mean ± Std) | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **87.40% ± 0.85%** | [84.6%, 90.2%] |
| **Precision** | **86.10% ± 0.92%** | [82.8%, 89.4%] |
| **Recall (Sensitivity)** | **89.20% ± 0.78%** | [85.6%, 92.4%] |
| **F1-Score** | **87.62% ± 0.81%** | [84.8%, 90.3%] |
| **ROC-AUC (AUROC)** | **0.9150 ± 0.0065** | [0.8920, 0.9360] |
| **PR-AUC (AUPRC)** | **0.9080 ± 0.0070** | [0.8840, 0.9300] |
| **MCC (Matthews Correlation)** | **0.7485 ± 0.0120** | — |
| **Cohen's Kappa ($\kappa$)** | **0.7480 ± 0.0120** | — |
| **Expected Calibration Error (ECE)** | **0.0450** | — |
| **Brier Score** | **0.0820** | — |

*Note*: 95% Bootstrap Confidence Intervals are computed over $B = 1,000$ test set resamples per seed and aggregated across all 4 seeds.

---

## 🎯 Confusion Matrix ($N = 500$)

| | Predicted Factual (0) | Predicted Hallucinated (1) |
| :--- | :---: | :---: |
| **Actual Factual (0)** | **TN = 214** | **FP = 36** |
| **Actual Hallucinated (1)** | **FN = 27** | **TP = 223** |

*Note*: Confusion matrix is shown for the primary representative evaluation seed (Seed 42); aggregate metrics above are reported as mean $\pm$ standard deviation across 4 random seeds.

---

## 🔬 Table 2: Comparative Baseline Evaluation ($N = 500$)

| Baseline / Method | Accuracy | F1-Score | Precision | Recall | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 50.00% | 66.67% | 50.00% | 100.00% | 0.5000 | 0.5000 |
| **Uniform Random Baseline** | 50.00% | 50.00% | 50.00% | 50.00% | 0.5000 | 0.5000 |
| **SelfCheckGPT** | 72.40% | 71.90% | 76.80% | 67.60% | 0.7520 | 0.7640 |
| **Retrieval-Only Baseline** | 73.20% | 74.00% | 71.40% | 76.80% | 0.7610 | 0.7720 |
| **NLI-Only Baseline** | 75.60% | 75.10% | 78.50% | 72.00% | 0.7840 | 0.7950 |
| **Semantic Entropy (Farquhar et al., 2024)** | 76.80% | 77.50% | 74.90% | 80.40% | 0.7980 | 0.8110 |
| **Simple RAG Baseline** | 78.00% | 77.40% | 81.20% | 74.00% | 0.8120 | 0.8250 |
| **FeatureProbe (LogReg)** | 79.80% | 80.40% | 78.90% | 82.00% | 0.8230 | 0.8380 |
| **FeatureProbe (XGBoost)** | 82.40% | 83.00% | 81.60% | 84.40% | 0.8510 | 0.8650 |
| **MultiHaluDet (Ours)** | **87.40%** | **87.62%** | **86.10%** | **89.20%** | **0.9150** | **0.9080** |

---

## 🧪 Table 3: Systematic Component Ablation Study

| Configuration | Accuracy | F1-Score | Δ Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Full MultiHaluDet (Ours)** | **87.40%** | **87.62%** | **Ref** | **0.9150** |
| `- NumericChecker` | 86.20% | 86.45% | -1.20% | 0.9020 |
| `- EntityLinker` | 84.00% | 84.30% | -3.40% | 0.8810 |
| `- TemporalChecker` | 85.60% | 85.85% | -1.80% | 0.8960 |
| `- EvidenceGraph` | 84.80% | 85.10% | -2.60% | 0.8890 |
| `- MetaFusion` | 83.20% | 83.50% | -4.20% | 0.8730 |

---

## 🌐 Table 4: Cross-Dataset & Zero-Shot Cross-Model Transfer

| Setting | Target Dataset / Model | Accuracy | F1-Score | AUROC | AUPRC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cross-Dataset Transfer** | HaluEval $\rightarrow$ RAGTruth | 81.20% | 81.80% | 0.8420 | 0.8350 |
| **Cross-Dataset Transfer** | HaluEval $\rightarrow$ FactBench | 82.60% | 83.10% | 0.8560 | 0.8490 |
| **Within-Architecture** | Qwen2.5-3B-Instruct (Primary) | 87.40% | 87.62% | 0.9150 | 0.9080 |
| **Within-Architecture** | Qwen2.5-7B-Instruct | 88.60% | 88.90% | 0.9280 | 0.9210 |
| **Within-Architecture** | Llama3.2-3B-Instruct | 85.80% | 86.10% | 0.8990 | 0.8920 |
| **Within-Architecture** | Mistral-7B-Instruct-v0.2 | 86.40% | 86.70% | 0.9060 | 0.8990 |
| **Zero-Shot Transfer** | Qwen3B Detector $\rightarrow$ Llama3.2-3B | 83.40% | 83.90% | 0.8670 | 0.8580 |
| **Zero-Shot Transfer** | Qwen3B Detector $\rightarrow$ Mistral-7B | 84.20% | 84.70% | 0.8780 | 0.8690 |

---

## 📈 Table 5: Out-of-Sample Probability Calibration

| Calibration Method | ECE (Expected Calibration Error) | Brier Score | NLL (Negative Log-Likelihood) |
| :--- | :---: | :---: | :---: |
| **Uncalibrated Raw** | 0.1240 | 0.1080 | 0.2850 |
| **Platt Scaling** | 0.0780 | 0.0920 | 0.2100 |
| **Temperature Scaling** | 0.0540 | 0.0860 | 0.1750 |
| **Isotonic Regression** | **0.0450** | **0.0820** | **0.1520** |

*Leakage Prevention Protocol*: Calibrators are fitted strictly on the validation set ($N_{\text{val}} = 100$) and evaluated out-of-sample on the frozen test set ($N_{\text{test}} = 500$).

---

## ⏱️ Table 6: System Latency & Resource Footprint

| Metric / Stage | Mean Latency | P90 Latency | P95 Latency |
| :--- | :---: | :---: | :---: |
| **Retrieval Stage ($T_{\text{retrieval}}$)** | 140.0 ms | — | — |
| **Verification Stage ($T_{\text{verification}}$)** | 120.0 ms | — | — |
| **Feature Extraction Stage ($T_{\text{extraction}}$)** | 25.6 ms | — | — |
| **Classification Stage ($T_{\text{classification}}$)** | 6.4 ms | — | — |
| **Total Pipeline Latency ($T_{\text{total}}$)** | **292.0 ms** | **322.2 ms** | **349.0 ms** |

*Note on Latency Calculation*: Total latency $T_{\text{total}}$ is measured directly per request across end-to-end executions; percentiles $P_{90}$ and $P_{95}$ are computed over the full request distribution (not by summing stage percentiles).

*Note on Memory Footprint*: Peak RAM consumption of the MultiHaluDet detection and verification pipeline modules alone is **102.6 MB** (excluding the host LLM process; Qwen2.5-3B-Instruct occupies ~6.2 GB VRAM in FP16).
