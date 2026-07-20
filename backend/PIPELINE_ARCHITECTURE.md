# Pipeline Architecture Overview

## System Design

The HALLUCINATION_GUARD backend implements an 8-stage pipeline for real-time hallucination detection with WebSocket support for live progress tracking.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                            │
│  (main.py: Routes & WebSocket management)                   │
└────────────┬────────────────────────────────────────────────┘
             │
             ├→ [1] HTTP API Endpoints
             │     ├─ POST /api/analyze (create job)
             │     ├─ GET /api/job/{id} (status)
             │     ├─ GET /api/result/{id} (results)
             │     └─ DELETE /api/job/{id} (cancel)
             │
             ├→ [2] WebSocket Server
             │     └─ WS /ws/progress/{job_id} (live updates)
             │
             └→ [3] Job Processing Pipeline
                   │
                   ├→ Job Queue (async)
                   │     (queue_manager.py)
                   │
                   └→ Pipeline Service (pipeline_service.py)
                         │
                         ├→ Stage 1: Input Received
                         ├→ Stage 2: Generating Response
                         ├→ Stage 3: Hidden State Extraction
                         ├→ Stage 4: Feature Extraction
                         ├→ Stage 5: Hallucination Detection
                         ├→ Stage 6: Fact Verification (RAG)
                         ├→ Stage 7: Explainability
                         └→ Stage 8: Analysis Completed
                               │
                               └→ Persist to Database
                                   (SQLite/PostgreSQL)
```

---

## Component Responsibilities

### 1. **main.py** – FastAPI Application
- Defines HTTP API endpoints
- Manages WebSocket connections
- Handles middleware (CORS, rate limiting)
- Manages app lifecycle (startup/shutdown)

### 2. **job_manager.py** – Job Lifecycle Management
- Creates new analysis jobs
- Fetches job status
- Updates job state
- Handles job cancellation
- Manages retry counting

### 3. **queue_manager.py** – Async Job Queue
- En­queues jobs for processing
- Manages single worker async loop
- Executes pipeline for each job
- Broadcasts progress via WebSocket
- Handles job cleanup

### 4. **pipeline_service.py** – Pipeline Orchestration
- Executes 8-stage pipeline in sequence
- Manages inter-stage data flow
- Implements retry logic per stage
- Emits progress events to WebSocket
- Persists final results to database

### 5. **websocket_manager.py** – Connection Management
- Tracks active WebSocket connections by job_id
- Broadcasts stage events to all subscribers
- Caches latest event for late subscribers
- Handles connection lifecycle

### 6. **database.py** – Data Persistence
- Creates database session
- Initializes schema on startup
- Supports SQLite (dev) and PostgreSQL (prod)

---

## Data Flow

### Job Creation Flow
```
Client Request
    ↓
POST /api/analyze
    ↓
Validate Input
    ↓
Create Job (JobManager)
    ↓
Save to Database
    ↓
Enqueue (JobQueueManager)
    ↓
Return job_id to Client
```

### Pipeline Execution Flow
```
Dequeue Job
    ↓
Update Status → "running"
    ↓
FOR each Stage 1-8:
    ├─ Start stage timer
    ├─ Execute stage handler
    ├─ Emit progress event
    ├─ Update database
    ├─ Retry on failure (max 3 times)
    ├─ Emit error if all retries fail
    └─ Continue to next stage
    ↓
Persist Result
    ↓
Update Status → "completed"
    ↓
Emit final result event
```

### WebSocket Subscription Flow
```
WS /ws/progress/{job_id}
    ↓
ProgressWebSocketManager.connect()
    ├─ Accept connection
    ├─ Send cached event if exists
    └─ Register connection
    ↓
Pipeline emits events
    ↓
ProgressWebSocketManager.broadcast()
    └─ Send to all subscribers
    ↓
Client receives message
    ↓
ws.onmessage()
```

---

## Stage Execution Details

### Stage 1: Input Received
**Purpose:** Validate and classify input

**Input:** `input_text` or `input_image_path`

**Process:**
- Determine input type (text, image, or text_image)
- Validate input constraints
- Log input metadata

**Output:** `{ input_received: true, input_type: "text"|"image"|"text_image" }`

---

### Stage 2: Generating Response
**Purpose:** Generate response using LLM

**Input:** `input_text`, `input_image_path`

**Process:**
- Load LLM model (Llama-3 or Qwen-VL-Chat)
- Generate response from input
- Cache for later stages

**Output:** `{ generated_response: "model output..." }`

---

### Stage 3: Hidden State Extraction
**Purpose:** Extract internal representations

**Input:** Generated response, attention weights

**Process:**
- Extract token embeddings
- Capture attention maps
- Retrieve layer outputs

**Output:**
```json
{
  "token_embeddings": [0.12, 0.34, 0.56],
  "attention_maps": [0.11, 0.22, 0.33]
}
```

---

### Stage 4: Feature Extraction
**Purpose:** Extract multi-scale features

**Input:** Hidden states, token embeddings

**Process:**
- Apply dynamic layer sampling
- Compute multi-scale attention features
- Encoder feature extraction
- Self-attention pooling

**Output:**
```json
{
  "dynamic_layer_sampling": [0.2, 0.5, 0.7],
  "multi_scale_attention": [0.4, 0.6, 0.8],
  "transformer_encoder": [0.1, 0.3, 0.9],
  "self_attention_pooling": [0.25, 0.45, 0.65]
}
```

---

### Stage 5: Hallucination Detection
**Purpose:** Detect hallucination probability

**Input:** Extracted features

**Process:**
- Run ensemble classifier:
  - Random Forest
  - XGBoost
  - LightGBM
  - Logistic Regression
  - SVM
- Aggregate predictions via stacking
- Compute confidence score

**Output:**
```json
{
  "prediction": true,
  "probability": 0.942,
  "confidence": 94.2,
  "model_votes": {
    "random_forest": true,
    "xgboost": true,
    "lightgbm": true,
    "logistic_regression": true,
    "svm": true
  }
}
```

---

### Stage 6: Fact Verification (RAG)
**Purpose:** Retrieve and verify facts

**Input:** Generated response, input

**Process:**
- Query knowledge base (Wikipedia, FEVER)
- Retrieve supporting documents
- Extract contradicting claims
- Score relevance

**Output:**
```json
{
  "sources": ["Wikipedia", "FEVER Dataset"],
  "supporting_documents": ["Wikipedia article on France"],
  "contradictions": ["Paris is the capital of Germany is false"]
}
```

---

### Stage 7: Explainability
**Purpose:** Generate explanations

**Input:** Features, predictions, evidence

**Process:**
- Compute SHAP values
- Identify important tokens
- Generate attention heatmap
- Create readable explanation

**Output:**
```json
{
  "shap_values": [0.41, 0.26, 0.18],
  "important_tokens": ["Paris", "capital", "Germany"],
  "attention_heatmap": "generated/heatmap.png",
  "explanation_text": "The response conflicts with retrieved evidence..."
}
```

---

### Stage 8: Analysis Completed
**Purpose:** Finalize results

**Input:** All stage outputs

**Process:**
- Aggregate all results
- Compute total processing time
- Format for client

**Output:** `{ analysis_completed: true }`

---

## Database Models

### Job Table
```sql
CREATE TABLE jobs (
  job_id STRING PRIMARY KEY,
  status STRING,           -- pending|running|completed|failed|cancelled
  input_type STRING,       -- text|image|text_image
  user_id STRING,
  input_text TEXT,
  input_image_path STRING,
  created_at DATETIME,
  started_at DATETIME,
  completed_at DATETIME,
  error_message TEXT,
  retry_count INTEGER
);
```

### Stage Table
```sql
CREATE TABLE stages (
  id INTEGER PRIMARY KEY,
  job_id STRING FOREIGN KEY,
  stage_number INTEGER,    -- 1-8
  name STRING,
  status STRING,           -- pending|running|completed|failed
  progress_percentage FLOAT,
  start_time DATETIME,
  end_time DATETIME,
  duration_ms INTEGER,
  error_message TEXT,
  metadata_json JSON
);
```

### Result Table
```sql
CREATE TABLE results (
  id STRING PRIMARY KEY,
  job_id STRING UNIQUE FOREIGN KEY,
  hallucination_score FLOAT,
  confidence FLOAT,
  is_hallucination STRING,  -- yes|no|uncertain
  generated_response TEXT,
  hidden_states JSON,
  extracted_features JSON,
  retrieved_evidence JSON,
  supporting_documents JSON,
  contradictions JSON,
  verified_answer TEXT,
  shap_explanation JSON,
  important_tokens JSON,
  attention_heatmap STRING,
  explanation_text TEXT,
  total_processing_time_ms FLOAT,
  created_at DATETIME
);
```

---

## Error Handling & Retry Logic

### Retry Strategy
- **Max Retries:** 3 per stage (configurable)
- **Backoff:** Exponential (0.5s, 1s, 2s)
- **Retry Conditions:** Transient errors only
- **Fatal Errors:** Invalid input, missing models, database errors

### Error Propagation
```
Stage Failure
    ↓
Log Error
    ↓
Retry? (attempt < max_retries)
    ├─ YES: Wait backoff → Retry
    └─ NO: Emit error event → Mark job FAILED
    ↓
Update Job Status
    ↓
Broadcast error to WebSocket
```

---

## Performance Considerations

### Concurrency
- Single async worker queue per instance
- Scale horizontal with load balancer for multi-worker
- Supports up to 1000 concurrent WebSocket connections

### Memory
- Pipeline state is in-memory (per job)
- Results persisted to database
- Old job records can be cleaned up (7-day default)

### Latency
- Average stage: 500-1500ms (configurable in dev_mode)
- Total pipeline: ~4-6 seconds
- WebSocket broadcast: <100ms per subscriber

---

## Configuration

Key configuration options in `config.py`:

```python
# Database
DATABASE_URL = "sqlite:///hallucination_guard.db"

# Job Queue
JOB_TIMEOUT_SECONDS = 300
JOB_MAX_RETRIES = 3
JOB_CLEANUP_DAYS = 7

# Development Mode
DEV_MODE = True
DELAY_SIMULATION_ENABLED = True
STAGE_DELAY_MIN_MS = 500
STAGE_DELAY_MAX_MS = 1500
DEBUG_LOGGING_ENABLED = False

# WebSocket
WS_HEARTBEAT_INTERVAL = 30
WS_MAX_CONNECTIONS = 1000

# Rate Limiting
RATE_LIMIT_RPM = 60
```

---

## Testing Architecture

The test suite is organized as:

```
tests/
├── test_api_endpoints.py          # API testing
├── test_pipeline_service.py        # Pipeline unit tests
├── test_websocket_manager.py       # WebSocket connection tests
├── test_job_cancel_and_concurrency.py  # Concurrency tests
├── test_pipeline_retries.py        # Retry logic tests
├── test_mock_data_generators.py    # Mock data validation
├── test_event_payloads.py          # Schema validation
├── conftest.py                     # Test fixtures
└── sample_event_payloads.json      # Example data
```

See specific test files for implementation details.
