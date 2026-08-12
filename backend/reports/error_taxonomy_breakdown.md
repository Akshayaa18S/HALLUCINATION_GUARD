# Error Taxonomy Distribution & Manual Verification Protocol (RQ7)

- **Inter-Annotator Agreement (Cohen's κ)**: `0.8824`
- **Percent Agreement (50 samples)**: `90.0%`

| Error Category | Total Samples | Detected (TP) | Missed (FN) | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Entity Hallucination | 29 | 16 | 1 | 0.9412 | 1.0000 |
| Numerical Hallucination | 29 | 14 | 1 | 0.9333 | 0.9333 |
| Temporal Hallucination | 29 | 19 | 1 | 0.9500 | 1.0000 |
| Relation Hallucination | 29 | 7 | 2 | 0.7778 | 1.0000 |
| Multi-Hop Reasoning Failure | 28 | 10 | 1 | 0.9091 | 1.0000 |
| Retrieval Failure | 28 | 20 | 0 | 1.0000 | 1.0000 |
| Unverifiable / Ambiguous Claim | 28 | 17 | 0 | 1.0000 | 1.0000 |