# Explainability Quality Evaluation Report (RQ7)

### 1. Attribution Faithfulness (Top-k Deletion vs Random-k Deletion)
| Top-k Tokens Deleted | ΔS Top-k | ΔS Random-k | Faithfulness Ratio (F_k) | Faithfulness Gain |
| :---: | :---: | :---: | :---: | :---: |
| Top-1 | 0.0793 | 0.0153 | 5.18x | +0.0640 |
| Top-3 | 0.2295 | 0.0450 | 5.10x | +0.1845 |
| Top-5 | 0.3537 | 0.0745 | 4.74x | +0.2791 |
| Top-10 | 0.5022 | 0.1474 | 3.41x | +0.3548 |

### 2. Explanation Consistency Across Seeds
- **Mean Attribution Cosine Similarity**: `0.9617`
- **Mean Spearman Rank Correlation (ρ)**: `-0.0031`

### 3. Evidence Sufficiency Alignment
- **Evidence Support Alignment Score**: `1.0000`