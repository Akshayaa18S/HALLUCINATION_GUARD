# Generalization Benchmark Results (Cross-Dataset, Within-Model & Zero-Shot Transfer)

| Evaluation Setting | Target Dataset / Model | ROC-AUC | AUPRC | F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| Cross-Dataset | HaluEval -> RAGTruth | 0.9972 | 0.9976 | 0.9474 |
| Cross-Dataset | HaluEval -> FactBench | 0.9976 | 0.9980 | 0.9519 |
| Within-Model Architecture | Qwen2.5-3B-Instruct (Primary) | 0.9985 | 0.9988 | 0.9671 |
| Within-Model Architecture | Qwen2.5-7B-Instruct | 0.9986 | 0.9988 | 0.9720 |
| Within-Model Architecture | Llama3.2-3B-Instruct | 0.9985 | 0.9988 | 0.9720 |
| Within-Model Architecture | Mistral-7B-Instruct-v0.2 | 0.9990 | 0.9992 | 0.9813 |
| Zero-Shot Cross-Model Transfer | Qwen3B Detector -> Llama3.2-3B Representations | 0.9977 | 0.9981 | 0.9366 |
| Zero-Shot Cross-Model Transfer | Qwen3B Detector -> Mistral-7B Representations | 0.9965 | 0.9972 | 0.9100 |