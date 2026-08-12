# Real-Time System Stage-by-Stage Latency & Resource Breakdown

| Stage Index | Stage Name | Avg Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | GPU VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Stage 1: Input Received & Validation | 2.44 | 2.41 | 3.07 | 3.09 | 0 |
| 2 | Stage 2: Model Inference & Hidden-State Trajectory Probing | 139.22 | 139.37 | 177.30 | 183.69 | 3200 |
| 3 | Stage 3: Hidden-State Trajectory Feature Extraction | 17.93 | 18.08 | 20.79 | 20.84 | 0 |
| 4 | Stage 4: MultiHaluDet Ensemble Inference | 11.94 | 12.05 | 14.77 | 14.81 | 3200 |
| 5 | Stage 5: Claim Extraction, NER & Coreference Resolution | 34.87 | 34.44 | 40.22 | 42.25 | 0 |
| 6 | Stage 6: Dual-Source RAG Evidence Retrieval | 85.56 | 84.61 | 109.42 | 115.01 | 0 |
| 7 | Stage 7: Dual-Signal Fusion & Calibration | 3.93 | 3.75 | 4.95 | 5.24 | 0 |
| 8 | Stage 8: Explainability (XAI) & Aggregation | 15.11 | 15.57 | 18.00 | 18.96 | 0 |

**Total End-to-End Latency**: 311.00 ms (~0.31 sec per response)