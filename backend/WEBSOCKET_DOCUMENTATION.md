# WebSocket Endpoint Documentation

## Overview
The HALLUCINATION_GUARD backend uses WebSocket for real-time event streaming during pipeline execution. Clients can subscribe to job progress updates and receive stage completion events as they occur.

## Endpoint

```
ws://localhost:8000/ws/progress/{job_id}
```

### Parameters
- `job_id` (string, required): The unique job identifier returned from the `/api/analyze` endpoint.

## Connection Lifecycle

### 1. Subscribe to Job Progress
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`);

ws.onopen = (event) => {
  console.log('Connected to job progress stream');
};
```

### 2. Receive Progress Updates
The WebSocket connection will emit messages as stages complete. Each message follows the unified WebSocket message format.

### 3. Handle Disconnection
```javascript
ws.onclose = (event) => {
  console.log('Disconnected from job progress stream');
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## Message Format

### Unified WebSocket Message (All Events)
```json
{
  "message_type": "stage_progress|result|error|heartbeat",
  "data": {},
  "timestamp": "2026-07-15T12:00:00.000Z"
}
```

## Message Types

### 1. Stage Progress (`stage_progress`)
Emitted when a pipeline stage progresses or completes.

**Example:**
```json
{
  "message_type": "stage_progress",
  "data": {
    "job_id": "uuid-string",
    "stage": 2,
    "name": "Generating Response",
    "status": "running",
    "progress_percentage": 20,
    "start_time": "2026-07-15T12:00:00Z",
    "end_time": null,
    "duration_ms": null,
    "metadata": {
      "model": "Llama-3",
      "generated_response": "Generated text..."
    },
    "error_message": null
  },
  "timestamp": "2026-07-15T12:00:01Z"
}
```

**Fields:**
- `job_id` (string): Job identifier
- `stage` (number 1-8): Stage number
- `name` (string): Stage name
- `status` (enum): `pending|running|completed|failed`
- `progress_percentage` (number 0-100): Overall progress
- `start_time` (ISO8601): When stage started
- `end_time` (ISO8601): When stage completed (null if running)
- `duration_ms` (number): Milliseconds taken (null if running)
- `metadata` (object): Stage-specific data
- `error_message` (string): Error details if failed

### 2. Final Result (`result`)
Emitted when all stages complete successfully.

**Example:**
```json
{
  "message_type": "result",
  "data": {
    "job_id": "uuid-string",
    "status": "completed",
    "hallucination": true,
    "confidence": 0.942,
    "generated_response": "Generated response based on input",
    "verified_answer": "Verified correct answer",
    "retrieved_evidence": {
      "sources": ["Wikipedia"],
      "supporting_documents": ["Article about topic"]
    },
    "explanation": "The model prediction was...",
    "processing_time_ms": 4200.0
  },
  "timestamp": "2026-07-15T12:00:08Z"
}
```

**Fields:**
- `job_id` (string): Job identifier
- `status` (string): `completed` or `failed`
- `hallucination` (boolean): Whether hallucination detected
- `confidence` (number 0-1): Confidence score
- `generated_response` (string): Original LLM output
- `verified_answer` (string): Corrected answer
- `retrieved_evidence` (object): Evidence from RAG
- `explanation` (string): Explainability text
- `processing_time_ms` (number): Total time in milliseconds

### 3. Error (`error`)
Emitted when an error occurs during pipeline execution.

**Example:**
```json
{
  "message_type": "error",
  "data": {
    "job_id": "uuid-string",
    "stage": 3,
    "error_message": "Failed to extract hidden states",
    "timestamp": "2026-07-15T12:00:05Z"
  },
  "timestamp": "2026-07-15T12:00:05Z"
}
```

**Fields:**
- `job_id` (string): Job identifier
- `stage` (number): Stage where error occurred (nullable)
- `error_message` (string): Error description
- `timestamp` (ISO8601): When error occurred

### 4. Heartbeat (`heartbeat`)
Optional periodic heartbeat to keep connection alive.

**Example:**
```json
{
  "message_type": "heartbeat",
  "data": {
    "job_id": "uuid-string"
  },
  "timestamp": "2026-07-15T12:00:30Z"
}
```

## Stage Order

Pipeline executes in this order:

| Stage | Name | Purpose |
|-------|------|---------|
| 1 | Input Received | Validate and process input |
| 2 | Generating Response | Generate response using LLM |
| 3 | Hidden State Extraction | Extract internal representations |
| 4 | Feature Extraction | Extract multi-scale features |
| 5 | Hallucination Detection | Detect hallucination probability |
| 6 | Fact Verification | Retrieve and verify facts via RAG |
| 7 | Explainability | Generate explanations (SHAP, etc.) |
| 8 | Analysis Completed | Finalize results |

## Status Values

- `pending` – Stage queued, not started
- `running` – Stage currently executing
- `completed` – Stage finished successfully
- `failed` – Stage encountered an error

## Metadata by Stage

### Stage 1: Input Received
```json
{
  "input_received": true,
  "input_type": "text|image|text_image"
}
```

### Stage 2: Generating Response
```json
{
  "generated_response": "LLM output text..."
}
```

### Stage 3: Hidden State Extraction
```json
{
  "token_embeddings": [0.12, 0.34, 0.56],
  "attention_maps": [0.11, 0.22, 0.33]
}
```

### Stage 4: Feature Extraction
```json
{
  "dynamic_layer_sampling": [0.2, 0.5, 0.7],
  "multi_scale_attention": [0.4, 0.6, 0.8]
}
```

### Stage 5: Hallucination Detection
```json
{
  "prediction": true,
  "probability": 0.942,
  "confidence": 94.2,
  "model_votes": {
    "random_forest": true,
    "xgboost": true
  }
}
```

### Stage 6: Fact Verification
```json
{
  "sources": ["Wikipedia", "FEVER Dataset"],
  "supporting_documents": ["Document 1"],
  "contradictions": []
}
```

### Stage 7: Explainability
```json
{
  "shap_values": [0.41, 0.26, 0.18],
  "important_tokens": ["word1", "word2"],
  "explanation_text": "The model..."
}
```

## Error Handling

### Connection Lost
If the WebSocket connection is lost, reconnect and fetch the latest job status via `/api/job/{job_id}`.

### Late Subscription
If you subscribe after some stages have completed, the connection will immediately send the latest cached event for that job.

## Performance Considerations

- Each connection broadcasts to all WebSocket clients for that job
- Messages are sent as JSON text frames
- Connection timeout is configurable (default: 30 seconds)
- Maximum concurrent connections: 1000 (configurable)

## Examples

See `sample_event_payloads.json` for complete example payloads for all event types.
