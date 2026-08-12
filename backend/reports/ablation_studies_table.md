# MultiHaluDet Systematic Ablation Studies

## Layer Depth Sampling

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| 2 Layers | 0.9920 | 0.9912 | 0.8675 | -0.0016 |
| 4 Layers | 0.9944 | 0.9938 | 0.8810 | +0.0008 |
| 6 Layers (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |
| 8 Layers | 0.9944 | 0.9940 | 0.8941 | +0.0008 |
| All Layers | 0.9944 | 0.9939 | 0.9070 | +0.0008 |

## Multi-Scale Attention Pooling

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| No Attention Pooling | 0.9920 | 0.9912 | 0.8537 | -0.0016 |
| Single Scale [1] | 0.9940 | 0.9933 | 0.8810 | +0.0004 |
| Multi-Scale [1, 2] | 0.9944 | 0.9939 | 0.8810 | +0.0008 |
| Multi-Scale [1, 2, 4] (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |
| Multi-Scale [1, 2, 4, 8] | 0.9944 | 0.9940 | 0.8941 | +0.0008 |

## Feature Branch Combination

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Hidden States Only | 0.9920 | 0.9912 | 0.8810 | -0.0016 |
| Logits Only | 0.9908 | 0.9899 | 0.8395 | -0.0028 |
| Hidden + Logits | 0.9944 | 0.9939 | 0.8810 | +0.0008 |
| Hidden + Attention | 0.9944 | 0.9939 | 0.8810 | +0.0008 |
| Full (Hidden + Logits + Attn) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |

## Loss Function Composition

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| BCE Only | 0.9920 | 0.9912 | 0.8810 | -0.0016 |
| BCE + Focal | 0.9944 | 0.9938 | 0.8810 | +0.0008 |
| BCE + Focal + Asymmetric | 0.9944 | 0.9940 | 0.8810 | +0.0008 |
| BCE + Focal + Asymmetric + Contrastive (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |

## Ensemble Base Learner

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest Only | 0.9936 | 0.9929 | 0.8675 | +0.0000 |
| Gradient Boosting Only | 0.9940 | 0.9933 | 0.8810 | +0.0004 |
| XGBoost Only | 0.9944 | 0.9938 | 0.8810 | +0.0008 |
| LightGBM Only | 0.9944 | 0.9938 | 0.8810 | +0.0008 |
| Logistic Regression Only | 0.9920 | 0.9912 | 0.8537 | -0.0016 |
| 5-Member OOF Ensemble (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |

## Retrieval Pipeline Component

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Wikipedia Only | 0.9940 | 0.9933 | 0.8810 | +0.0004 |
| FEVER Only | 0.9936 | 0.9929 | 0.8675 | +0.0000 |
| BM25 Only | 0.9920 | 0.9912 | 0.8675 | -0.0016 |
| Dense Only | 0.9944 | 0.9938 | 0.8810 | +0.0008 |
| BM25 + Dense | 0.9944 | 0.9939 | 0.8810 | +0.0008 |
| BM25 + Dense + Reranker (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |

## Verification Mechanism

| Variant | ROC-AUC | AUPRC | F1 Score | Δ ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Cosine Similarity Only | 0.9920 | 0.9912 | 0.8810 | -0.0016 |
| NLI Only | 0.9944 | 0.9939 | 0.8810 | +0.0008 |
| Cosine + NLI (Full) | 0.9936 | 0.9931 | 0.8810 | +0.0000 |
