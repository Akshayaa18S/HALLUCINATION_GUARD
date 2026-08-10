# Research Evaluation Summary: HaluEval Benchmark Suite

## 1. Classification Metrics Summary

| Metric | Value |
| :--- | :---: |
| **Samples** | 6 |
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1 Score** | 1.0000 |
| **ROC-AUC** | 1.0000 |
| **False Positive Rate (FPR)** | 0.0000 |
| **False Negative Rate (FNR)** | 0.0000 |

## 2. Statistical Significance Testing (McNemar Test)

- **Baseline Accuracy**: 0.3333
- **Proposed Hybrid Accuracy**: 1.0000
- **Absolute Improvement**: +0.6667
- **McNemar p-value**: `0.13361`
- **Statistically Significant**: `NO`

## 3. Ablation Study Matrix

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (MultiHaluDet)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **+ Claim Verification** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **+ Entity Retrieval** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **+ Entity Disambiguation** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **+ Confidence Fusion** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **+ Full Hybrid Framework** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 4. Performance & Resource Benchmarks

| Stage / Resource | Benchmark Value |
| :--- | :---: |
| **Retrieval Latency** | 140.00 ms |
| **Verification Latency** | 120.00 ms |
| **Feature Extraction Latency** | 25.60 ms |
| **Classification Latency** | 6.40 ms |
| **Total Pipeline Latency** | 1469.00 ms |
| **Memory Usage** | 102.60 MB |
| **CPU Usage** | 21.20 % |

## 5. Confidence Calibration

- **Expected Calibration Error (ECE)**: `0.1133`
- **Brier Score**: `0.0140`

## 6. Error Analysis Taxonomy Breakdown

| Error Category | Count |
| :--- | :---: |
| **wrong_retrieval** | 0 |
| **entity_ambiguity** | 0 |
| **claim_extraction** | 0 |
| **verification** | 0 |
| **evidence_conflict** | 0 |
| **ensemble_disagreement** | 0 |
| **confidence_calibration** | 0 |
| **unknown** | 0 |
