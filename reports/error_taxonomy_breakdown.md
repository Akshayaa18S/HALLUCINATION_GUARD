# Error Taxonomy Distribution & Manual Verification Protocol (RQ7)

- **Inter-Annotator Agreement (Cohen's κ)**: `0.8824`
- **Percent Agreement (50 samples)**: `90.0%`

| Error Category | Total Samples | Detected (TP) | Missed (FN) | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Entity Hallucination | 72 | 13 | 24 | 0.3514 | 0.5000 |
| Numerical Hallucination | 72 | 11 | 25 | 0.3056 | 0.4231 |
| Temporal Hallucination | 72 | 9 | 26 | 0.2571 | 0.4091 |
| Relation Hallucination | 71 | 10 | 24 | 0.2941 | 0.4762 |
| Multi-Hop Reasoning Failure | 71 | 11 | 24 | 0.3143 | 0.4783 |
| Retrieval Failure | 71 | 13 | 23 | 0.3611 | 0.5200 |
| Unverifiable / Ambiguous Claim | 71 | 13 | 24 | 0.3514 | 0.4815 |