# Generalization Benchmark Results (Cross-Dataset, Within-Model & Zero-Shot Transfer)

| Evaluation Setting | Target Dataset / Model | ROC-AUC | AUPRC | F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| Cross-Dataset | HaluEval -> RAGTruth | 0.4496 | 0.4673 | 0.6667 |
| Cross-Dataset | HaluEval -> FactBench | 0.4599 | 0.4690 | 0.6667 |
| Within-Model Architecture | Qwen2.5-3B-Instruct (Primary) | 0.5155 | 0.5049 | 0.6667 |
| Within-Model Architecture | Qwen2.5-7B-Instruct | 0.5089 | 0.4997 | 0.6667 |
| Within-Model Architecture | Llama3.2-3B-Instruct | 0.4898 | 0.5049 | 0.6667 |
| Within-Model Architecture | Mistral-7B-Instruct-v0.2 | 0.4968 | 0.4971 | 0.6667 |
| Zero-Shot Cross-Model Transfer | Qwen3B Detector -> Llama3.2-3B Representations | 0.5329 | 0.5249 | 0.6667 |
| Zero-Shot Cross-Model Transfer | Qwen3B Detector -> Mistral-7B Representations | 0.5179 | 0.5151 | 0.6667 |