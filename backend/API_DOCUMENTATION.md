# HALLUCINATION_GUARD API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. For production, implement JWT or API key authentication.

---

## Endpoints

### Health Check

**GET** `/` 

**Description:** Check server health and configuration status

**Response (200):**
```json
{
  "status": "running",
  "app": "HALLUCINATION_GUARD",
  "version": "1.0.0",
  "debug": true,
  "dev_mode": true
}
```

---

### Submit Analysis Job

**POST** `/api/analyze`

**Description:** Submit text and/or image for hallucination detection analysis

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "input_text": "Is Paris the capital of Germany?",
  "input_image_path": "/path/to/image.jpg",
  "user_id": "user123"
}
```

**Parameters:**
- `input_text` (string, optional): Text to analyze
- `input_image_path` (string, optional): Path to image file
- `user_id` (string, optional): User identifier for tracking
- **Note:** At least one of `input_text` or `input_image_path` must be provided

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2026-07-15T12:00:00.000Z"
}
```

**Response (400):**
```json
{
  "error": "Either input_text or input_image_path must be provided",
  "status_code": 400
}
```

**Response (500):**
```json
{
  "error": "Failed to create analysis job: ...",
  "status_code": 500
}
```

---

### Get Job Status

**GET** `/api/job/{job_id}`

**Description:** Fetch current status and progress of a job

**Path Parameters:**
- `job_id` (string, required): Job identifier

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "input_type": "text",
  "progress_percentage": 45.0,
  "current_stage": 4,
  "current_stage_name": "Feature Extraction",
  "started_at": "2026-07-15T12:00:01.000Z",
  "created_at": "2026-07-15T12:00:00.000Z",
  "retry_count": 0
}
```

**Status Values:**
- `pending` – Waiting to execute
- `running` – Currently executing pipeline
- `completed` – Finished successfully
- `failed` – Encountered error
- `cancelled` – User cancelled the job

**Response (404):**
```json
{
  "error": "Job 550e8400-e29b-41d4-a716-446655440000 not found",
  "status_code": 404
}
```

---

### Get Analysis Result

**GET** `/api/result/{job_id}`

**Description:** Fetch final analysis results (available after job completion)

**Path Parameters:**
- `job_id` (string, required): Job identifier

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "hallucination_score": 94.2,
  "confidence": 0.942,
  "is_hallucination": "yes",
  "generated_response": "Paris is the capital of Germany.",
  "verified_answer": "Paris is the capital of France.",
  "retrieved_evidence": {
    "sources": ["Wikipedia"],
    "supporting_documents": ["France article"],
    "contradictions": ["Germany capital is Berlin"]
  },
  "explanation_text": "The model incorrectly identified Paris as German capital, contradicting retrieved knowledge.",
  "total_processing_time_ms": 4200.0,
  "created_at": "2026-07-15T12:00:08.000Z"
}
```

**Response (404):**
```json
{
  "error": "Result for job 550e8400-e29b-41d4-a716-446655440000 not found",
  "status_code": 404
}
```

---

### Cancel Job

**DELETE** `/api/job/{job_id}`

**Description:** Cancel a running or pending job

**Path Parameters:**
- `job_id` (string, required): Job identifier

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "cancelled_at": "2026-07-15T12:00:05.000Z"
}
```

**Response (404):**
```json
{
  "error": "Job 550e8400-e29b-41d4-a716-446655440000 not found",
  "status_code": 404
}
```

---

## WebSocket Streaming

**WS** `/ws/progress/{job_id}`

**Description:** Subscribe to real-time job progress events

See [WebSocket Documentation](./WEBSOCKET_DOCUMENTATION.md) for detailed message formats and examples.

### Quick Start
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`);

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(`Stage: ${message.data.name}, Progress: ${message.data.progress_percentage}%`);
};
```

---

## Rate Limiting

Rate limiting is applied to API endpoints:

**Rate Limit:** 60 requests per minute (configurable)

**Response Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
```

**Response (429 - Too Many Requests):**
```json
{
  "detail": "Rate limit exceeded"
}
```

**Headers:**
```
Retry-After: 30
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
```

---

## CORS

CORS is enabled for the following origins (configurable):
- `http://localhost:3000` – Frontend dev server
- `http://localhost:8000` – Backend dev server

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Human-readable error message",
  "status_code": 400
}
```

### Common Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |

---

## Workflow Example

```
1. POST /api/analyze
   Response: { job_id: "abc123", status: "pending" }

2. GET /api/job/abc123
   Response: { status: "running", progress_percentage: 45 }

3. WS /ws/progress/abc123
   Receive stage_progress events as pipeline executes

4. GET /api/result/abc123
   Response: { hallucination: true, confidence: 0.94, ... }
```

---

## Development Mode

In development mode (`DEV_MODE=true`), the backend includes:

- Stage delay simulation (0.5-1.5 seconds per stage)
- Mock LLM and RAG responses
- Debug logging (when enabled)
- Fake data generation for testing

Disable or configure these using environment variables:

```bash
DELAY_SIMULATION_ENABLED=false
DEBUG_LOGGING_ENABLED=true
MOCK_DATA_MODE=true
```

---

## Testing

Use the provided test files and sample payloads:

- `tests/test_api_endpoints.py` – API endpoint tests
- `tests/sample_event_payloads.json` – Example WebSocket messages
- `utils/mock_data.py` – Mock data generators

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review WebSocket endpoint documentation
3. Run tests to validate setup
