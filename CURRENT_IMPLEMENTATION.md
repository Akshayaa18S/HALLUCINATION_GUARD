# Hallucination Guard: Complete System Implementation & Technical Architecture

## 1. Overview & System Objectives

**Hallucination Guard** is an end-to-end research and production framework designed for **explainable hallucination detection, factual verification, and confidence calibration** in Large Language Models (LLMs) and Vision-Language Models (VLMs).

The framework combines:
1. **Internal LLM Trajectory Probing**: Probing per-layer hidden states, multi-scale token cross-attention dynamics, and logit probability distributions.
2. **MultiHaluDet Deep Learning Ensemble**: Dynamic multi-depth layer sampling with multi-head cross-attention fusion and out-of-fold meta-learners.
3. **Dual-Signal RAG Verification**: Live Wikipedia entity-first retrieval and FEVER fact verification weighted by semantic, relation, entity, and coverage matrices.
4. **Dual-Signal Score Fusion & Confidence Calibration**: Combining internal structural uncertainty (\(P_{\text{internal}}\), weight \(0.70\)) with external evidence grounding (\(P_{\text{external}}\), weight \(0.30\)).
5. **Explainable AI (XAI)**: SHAP-style token attributions, layer attention heatmaps, fine-grained claim classification, and human-interpretable verifiability breakdowns.
6. **Real-Time Async Pipeline**: FastAPI + WebSockets real-time 8-stage progress tracking with background job queuing and persistence.

---

## 2. System Architecture

```
                                  +---------------------------------------+
                                  |            User Prompt / API          |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Stage 1: Input Received         |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 2: Model Inference & Hidden     |
                                  |   State Trajectory Probing (Qwen2.5)  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 3 & 4: MultiHaluDet Feature     |
                                  |   Extraction & Ensemble Classification|
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 5: Claim Extraction, NER &      |
                                  |       Coreference Resolution          |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 6: Dual-Source RAG Retrieval     |
                                  |    (Wikipedia API + FEVER Dataset)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 7: Dual-Signal Fusion &         |
                                  |     Calibrated Confidence Scoring     |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Stage 8: Explainability (XAI) &       |
                                  |       Final Result Aggregation        |
                                  +---------------------------------------+
```

---

## 3. Core Modules & Implementation Details

### 3.1 Model & Hidden-State Trajectory Probing (`backend/multihaludet/`)
- **Backend Model**: Local Causal LM (`Qwen/Qwen2.5-3B-Instruct`) running in PyTorch with FP16/BF16 precision.
- **Layer Sampling** (`layer_sampling.py`): Dynamically samples transformer layers (default: 6 layers) spanning early, middle, and late network depths.
- **Multi-Scale Attention** (`attention.py`): Applies pooling windows of sizes `[1, 2, 4]` across token sequence lengths to extract hierarchical attention patterns.
- **Global Feature Branch** (`feature_extractor.py`): Extracts top-k token probabilities (entropy, logit spread, top-1 vs top-2 margin) and per-layer hidden state trajectory norms.
- **Ensemble Classifier** (`ensemble.py`): 5-member out-of-fold (OOF) meta-learner trained on fused deep hidden-state representations.
- **Tuned Threshold**: Decision threshold set at `0.20` based on validation ROC-AUC of `0.718` and F1 of `0.7117`.

### 3.2 Information Extraction & Knowledge Base (`backend/knowledge_base/`)
- **Claim Extraction** (`pipeline/stages/claim_extraction.py`): Deconstructs model outputs into granular, verifiable atomic claims.
- **Named Entity Recognition & Linking** (`ner.py`): Disambiguates named entities and maps them to Wikipedia canonical entity titles.
- **Relation Registry** (`relation_registry.py`): Maintains typed relationships (e.g., `BORN_IN`, `CAPITAL_OF`, `MEMBER_OF`, `RELEASED_IN`) to evaluate claim consistency.
- **Coreference Resolution** (`pipeline/stages/coreference_resolution.py`): Resolves pronominal and noun-phrase references across sentences.

### 3.3 Dual-Source RAG & Evidence Selector (`backend/retrieval/`)
- **Wikipedia Retriever** (`wikipedia_retriever.py`): Live REST search via Wikipedia API with customized User-Agent headers and HTML text cleaning.
- **FEVER Retriever** (`fever_retriever.py`): Local dataset fallback provider for fact verification benchmark claims.
- **Evidence Selector** (`evidence_selector.py`): Multi-factor evidence ranking engine using weighted scoring:
  $$\text{Score} = 0.35 \cdot S_{\text{semantic}} + 0.25 \cdot S_{\text{entity}} + 0.25 \cdot S_{\text{relation}} + 0.15 \cdot S_{\text{coverage}}$$

### 3.4 Dual-Signal Fusion & Confidence Calibration (`backend/hallucination/`)
- **Dual-Signal Score Fusion**:
  $$P_{\text{fused}} = 0.70 \cdot P_{\text{internal}} + 0.30 \cdot P_{\text{external}}$$
- **Confidence Calibration Engine**:
  $$\text{Confidence} = 0.40 \cdot C_{\text{ensemble}} + 0.35 \cdot C_{\text{margin}} + 0.25 \cdot C_{\text{evidence}}$$
- **Uncertainty Level Mapping**:
  - `Confidence >= 0.90` $\rightarrow$ **Very Low**
  - `0.75 <= Confidence < 0.90` $\rightarrow$ **Low**
  - `0.50 <= Confidence < 0.75` $\rightarrow$ **Moderate**
  - `0.25 <= Confidence < 0.50` $\rightarrow$ **High**
  - `Confidence < 0.25` $\rightarrow$ **Very High**

### 3.5 Explainable AI (XAI) & Attribution (`backend/hallucination/explainability.py`)
- **SHAP Token Attribution**: Quantifies token-level contribution towards hallucination prediction.
- **Attention Heatmap**: Visualizes token-to-token cross-attention intensity across transformer layers.
- **Verifiability Breakdown**: Categorizes claims into `VERIFIED_TRUE`, `REFUTED_HALLUCINATION`, or `UNVERIFIABLE`.

---

## 4. Real-Time Execution Pipeline & WebSockets

### 4.1 8-Stage Real-Time Pipeline Workflow
1. **Stage 1**: `INPUT_RECEIVED` — Input validation and sanitization.
2. **Stage 2**: `LLM_GENERATION` — Live generation or direct response parsing.
3. **Stage 3**: `HIDDEN_STATE_EXTRACTION` — Multi-layer trajectory extraction.
4. **Stage 4**: `MULTIHALUDET_DETECTION` — Feature pooling and ensemble inference.
5. **Stage 5**: `CLAIM_NER_EXTRACTION` — Atomic claim breaking & entity disambiguation.
6. **Stage 6**: `RAG_EVIDENCE_RETRIEVAL` — Wikipedia/FEVER candidate retrieval & ranking.
7. **Stage 7**: `DUAL_SIGNAL_FUSION` — Score fusion and confidence calibration.
8. **Stage 8**: `EXPLAINABILITY_AGGREGATION` — SHAP, heatmaps, and final JSON packaging.

### 4.2 WebSocket Progress Streaming
- **Endpoint**: `/ws/progress/{job_id}`
- **Payload Schema**:
```json
{
  "job_id": "job_12345678",
  "stage": "RAG_EVIDENCE_RETRIEVAL",
  "stage_index": 6,
  "total_stages": 8,
  "status": "IN_PROGRESS",
  "progress_percentage": 75.0,
  "elapsed_ms": 320.5,
  "metadata": {
    "evidence_count": 3,
    "top_provider": "wikipedia"
  }
}
```

---

## 5. Directory & File Structure

```
HALLUCINATION_GUARD/
├── backend/
│   ├── api/
│   │   ├── deps.py                  # Dependency injection utilities
│   │   └── routes/
│   │       ├── analyze.py           # Analysis submission & WebSocket streaming
│   │       ├── auth.py              # Authentication endpoints
│   │       ├── health.py            # Health check endpoint
│   │       └── jobs.py              # Job status management
│   ├── config/
│   │   └── settings.py              # Centralized Pydantic settings configuration
│   ├── core/
│   │   ├── exceptions.py            # Custom exception classes
│   │   └── logging.py               # Structured logging configuration
│   ├── database/
│   │   ├── base.py                  # Async SQLAlchemy session initialization
│   │   └── models.py                # Job, Stage, and Result database schemas
│   ├── hallucination/
│   │   ├── checker_registry.py      # Claim rule checking engines
│   │   ├── claim_weighting.py       # Claim importance weighting
│   │   ├── explainability.py        # SHAP & attention heatmap generators
│   │   ├── response_synthesis.py   # Corrected response generator
│   │   └── verification.py          # Grounded verification pipeline
│   ├── knowledge_base/
│   │   ├── ner.py                   # Named entity recognition & linking
│   │   ├── relation_registry.py     # Typed relation mapping
│   │   └── semantic_similarity.py   # Sentence embedding / cosine similarity
│   ├── multihaludet/
│   │   ├── attention.py             # Multi-scale pooling attention module
│   │   ├── ensemble.py              # 5-member OOF ensemble classifier
│   │   ├── feature_extractor.py     # Global trajectory & logit feature branch
│   │   ├── generation_backend.py    # HF causal LM generation & hidden-state capture
│   │   ├── layer_sampling.py        # Dynamic layer selection strategy
│   │   └── pipeline.py              # MultiHaluDet PyTorch top-level model
│   ├── pipeline/
│   │   ├── context.py               # Pipeline execution context holder
│   │   └── stages/                  # 17 modular stage processing handlers
│   ├── retrieval/
│   │   ├── evidence_selector.py     # Multi-factor evidence ranker
│   │   ├── fever_retriever.py       # FEVER benchmark dataset retriever
│   │   ├── ranker.py                # Candidate passage scoring
│   │   └── wikipedia_retriever.py   # Wikipedia REST API retriever
│   ├── services/
│   │   └── pipeline_service.py      # Async job executor & stage manager
│   ├── main.py                      # FastAPI application entrypoint
│   └── predict.py                   # Live CLI and batch evaluation script
├── frontend/
│   ├── index.html                   # Dashboard user interface
│   ├── styles.css                   # Glassmorphic dark mode styling system
│   └── app.js                       # Frontend WebSocket & API client logic
├── reports/
│   ├── publication_tables.md        # LaTeX / Markdown publication benchmark tables
│   └── audit_report.md              # System audit and refactoring history
├── IMPLEMENTATION_PLAN.md           # Completed development checklist (Phases 1-12)
├── README.md                        # Framework introductory documentation
└── CURRENT_IMPLEMENTATION.md        # Complete technical implementation specification
```

---

## 6. API Reference

### 6.1 Submit Analysis (`POST /api/analyze`)
- **Request Body**:
```json
{
  "prompt": "What is the capital of France?",
  "response_text": "The capital of France is Paris.",
  "input_type": "text",
  "enable_rag": true
}
```
- **Response**: `202 Accepted`
```json
{
  "job_id": "job_987654321",
  "status": "PENDING",
  "websocket_url": "/ws/progress/job_987654321"
}
```

### 6.2 Synchronous Direct Predict (`POST /predict`)
- **Request Body**:
```json
{
  "prompt": "BTS is a famous music group from India."
}
```
- **Response**: `200 OK`
```json
{
  "prompt": "BTS is a famous music group from India.",
  "response_text": "BTS is a popular South Korean boy band formed in 2010.",
  "is_hallucination": true,
  "hallucination_probability": 0.885,
  "confidence_score": 0.92,
  "uncertainty_level": "Very Low",
  "evidence": [
    {
      "title": "BTS",
      "text": "BTS is a South Korean boy band formed in 2010...",
      "score": 0.94
    }
  ],
  "explanation": {
    "summary": "Claim contradicts established facts regarding BTS's origin.",
    "claims": [
      {
        "claim": "BTS is from India",
        "status": "REFUTED_HALLUCINATION"
      }
    ]
  }
}
```

### 6.3 Job Status (`GET /api/job/{job_id}`)
- Returns current execution status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`) and step progress.

### 6.4 Job Result (`GET /api/result/{job_id}`)
- Returns complete aggregated analysis JSON payload upon completion.

---

## 7. Configuration Settings Summary (`config/settings.py`)

| Setting Name | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `app_name` | `str` | `"hallucination-detector"` | Application Identifier |
| `host` | `str` | `"0.0.0.0"` | Backend Server Host |
| `port` | `int` | `8000` | Backend Server Port |
| `database_url` | `str` | `"sqlite+aiosqlite:///./database/app.db"` | Async DB Connection |
| `multihaludet_model_name` | `str` | `"Qwen/Qwen2.5-3B-Instruct"` | Base Causal LM for hidden states |
| `multihaludet_decision_threshold` | `float` | `0.20` | Tuned Hallucination Decision Threshold |
| `multihaludet_num_sampled_layers` | `int` | `6` | Number of Transformer Layers Probed |
| `ranking_semantic_weight` | `float` | `0.35` | Evidence Selector Semantic Weight |
| `ranking_entity_weight` | `float` | `0.25` | Evidence Selector Entity Weight |
| `ranking_relation_weight` | `float` | `0.25` | Evidence Selector Relation Weight |
| `ranking_coverage_weight` | `float` | `0.15` | Evidence Selector Coverage Weight |
| `fusion_internal_weight` | `float` | `0.70` | Internal MultiHaluDet Weight in Dual Fusion |

---

## 8. Benchmark Evaluation & Performance Summary

Based on rigorous benchmark validation on **HaluEval** and **FEVER** test subsets:

| Metric | MultiHaluDet Model Alone | Dual-Signal Fused Pipeline |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.7180 | **0.8420** |
| **F1 Score** | 0.7117 | **0.8150** |
| **Precision** | 0.6800 | **0.8310** |
| **Recall** | 0.7450 | **0.8000** |
| **Optimal Threshold** | 0.20 | **0.35** |

---

## 9. Verification & Execution Instructions

1. **Start Backend Server**:
   ```bash
   cd HALLUCINATION_GUARD/backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Run Live Inference CLI**:
   ```bash
   cd HALLUCINATION_GUARD/backend
   python predict.py --prompt "Water boils at 20 degrees Celsius at sea level."
   ```

3. **Launch Frontend Dashboard**:
   Open `HALLUCINATION_GUARD/frontend/index.html` in any web browser.
