# Publication Baseline Comparison Table

| Baseline Method | ROC-AUC | AUPRC | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Majority_Class | 0.5000 | 0.5000 | 0.6667 | 0.5000 | 1.0000 |
| Uniform_Random | 0.2500 | 0.5000 | 0.4000 | 0.3333 | 0.5000 |
| FeatureProbe_LogReg | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 0.5000 |
| FeatureProbe_XGBoost | 0.5000 | 0.5000 | 0.6667 | 0.5000 | 1.0000 |
| Simple_Hidden_Probe | 0.7500 | 0.8333 | 0.6667 | 1.0000 | 0.5000 |
| SelfCheckGPT | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Semantic_Entropy | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 1.0000 |
| Retrieval_Only | 0.5000 | 0.5000 | 0.6667 | 0.5000 | 1.0000 |
| NLI_Only | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Retrieval_Plus_NLI | 0.5000 | 0.5000 | 0.6667 | 0.5000 | 1.0000 |