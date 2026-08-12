# MultiHaluDet Systematic Ablation Studies

## Layer Depth Sampling

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| 2 Layers | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| 4 Layers | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 6 Layers (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 8 Layers | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| All Layers | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Multi-Scale Attention Pooling

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| No Attention Pooling | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| Single Scale [1] | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Multi-Scale [1, 2] | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Multi-Scale [1, 2, 4] (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Multi-Scale [1, 2, 4, 8] | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Feature Branch Combination

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Hidden States Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Logits Only | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| Hidden + Logits | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Hidden + Attention | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Full (Hidden + Logits + Attn) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Loss Function Composition

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| BCE Only | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| BCE + Focal | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| BCE + Focal + Asymmetric | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| BCE + Focal + Asymmetric + Contrastive (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Ensemble Base Learner

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Gradient Boosting Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| XGBoost Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| LightGBM Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Logistic Regression Only | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| 5-Member OOF Ensemble (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Retrieval Pipeline Component

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Wikipedia Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| FEVER Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| BM25 Only | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| Dense Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| BM25 + Dense | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| BM25 + Dense + Reranker (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |

## Verification Mechanism

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Cosine Similarity Only | 1.0000 | 1.0000 | 0.9873 | +0.0000 |
| NLI Only | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Cosine + NLI (Full) | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
