# Publication Baseline Comparison Table

| Baseline Method | ROC-AUC | AUPRC | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Majority_Class | 0.5000 | 0.4700 | 0.0000 | 0.0000 | 0.0000 |
| Uniform_Random | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FeatureProbe_LogReg | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FeatureProbe_XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Simple_Hidden_Probe | 0.9342 | 0.9194 | 0.8602 | 0.8696 | 0.8511 |
| SelfCheckGPT | 0.5000 | 0.4700 | 0.0000 | 0.0000 | 0.0000 |
| Semantic_Entropy | 0.5022 | 0.5006 | 0.6395 | 0.4700 | 1.0000 |
| Retrieval_Only | 0.5000 | 0.4700 | 0.0000 | 0.0000 | 0.0000 |
| NLI_Only | 0.5000 | 0.4700 | 0.6395 | 0.4700 | 1.0000 |
| Retrieval_Plus_NLI | 0.5000 | 0.4700 | 0.0000 | 0.0000 | 0.0000 |