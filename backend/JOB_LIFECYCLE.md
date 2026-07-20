# Job Lifecycle & State Transitions

## Job States

A job progresses through the following states during its lifetime:

```
┌─────────┐
│ pending │  ← Job created, waiting to execute
└────┬────┘
     │
     ↓
┌─────────┐     ┌──────────┐
│ running │────→│  failed  │  (error during execution)
└────┬────┘     └──────────┘
     │
     ├─→ ┌──────────────┐
     │   │  completed   │  (successfully finished all stages)
     │   └──────────────┘
     │
     └─→ ┌──────────────┐
         │  cancelled   │  (user requested cancellation)
         └──────────────┘
```

---

## Detailed State Flowchart

### 1. Job Creation (PENDING)

**Triggered by:** `POST /api/analyze`

**Actions:**
- Generate unique job_id
- Save job metadata to database
- Set status = "pending"
- Record created_at timestamp
- Enqueue for processing

**Transition:** → RUNNING (when worker picks up job)

**Example:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "pending",
  "created_at": "2026-07-15T12:00:00Z",
  "input_type": "text",
  "retry_count": 0
}
```

---

### 2. Execution (RUNNING)

**Triggered by:** Job queue worker dequeues job

**Actions:**
- Set status = "running"
- Record started_at timestamp
- Begin pipeline execution (stages 1-8)
- Emit progress events via WebSocket
- Handle per-stage retries

**Transitions:**
- → COMPLETED (all stages succeed)
- → FAILED (stage fails after retries exhausted)
- → CANCELLED (user requests cancellation)

**Example:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "running",
  "progress_percentage": 35.0,
  "current_stage": 3,
  "current_stage_name": "Hidden State Extraction",
  "started_at": "2026-07-15T12:00:01Z"
}
```

---

### 3. Success (COMPLETED)

**Triggered by:** Pipeline finishes stage 8 successfully

**Actions:**
- Set status = "completed"
- Record completed_at timestamp
- Save Result object with all analysis data
- Emit final result event
- Broadcast results to WebSocket subscribers

**No further transitions**

**Example:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "completed",
  "completed_at": "2026-07-15T12:00:08Z",
  "hallucination": true,
  "confidence": 0.942,
  "processing_time_ms": 8000.0
}
```

---

### 4. Failure (FAILED)

**Triggered by:** Stage fails after max_retries exhausted

**Actions:**
- Set status = "failed"
- Record completed_at timestamp
- Store error_message
- Emit error event
- Broadcast error to WebSocket subscribers

**No further transitions** (can DELETE to clean up)

**Example:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "failed",
  "error_message": "Stage 3 failed: Unable to extract hidden states",
  "completed_at": "2026-07-15T12:00:03Z"
}
```

---

### 5. Cancellation (CANCELLED)

**Triggered by:** `DELETE /api/job/{job_id}`

**Actions:**
- Set status = "cancelled"
- Record completed_at timestamp
- Stop pipeline execution
- Broadcast cancellation event
- Clean up in-flight operations

**No further transitions**

**Example:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "cancelled",
  "cancelled_at": "2026-07-15T12:00:02Z"
}
```

---

## Retry Logic

### Per-Stage Retries

Each stage implements automatic retry logic:

```
Execute Stage
    ↓
Failure?
    ├─ NO: Continue
    └─ YES:
        ├─ attempt < max_retries?
        │   ├─ YES: Wait for backoff → Retry
        │   └─ NO: Fail stage → Emit error
        └─ Update database
```

### Configuration

```python
# config.py
JOB_MAX_RETRIES = 3  # attempts per stage
```

### Backoff Strategy

- **Attempt 1:** Failure logged
- **Attempt 2:** Wait 0.5s, retry
- **Attempt 3:** Wait 1.0s, retry
- **Attempt 4:** Wait 2.0s, retry (if max_retries >= 4)
- **Final Failure:** Mark stage FAILED, emit error

---

## Job Status Polling

### Get Current Status

**Request:**
```
GET /api/job/a1b2c3d4-e5f6-...
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-...",
  "status": "running",
  "input_type": "text",
  "progress_percentage": 45.0,
  "current_stage": 4,
  "current_stage_name": "Feature Extraction",
  "started_at": "2026-07-15T12:00:01Z",
  "created_at": "2026-07-15T12:00:00Z",
  "retry_count": 0
}
```

### Available Fields

| Field | Description |
|-------|-------------|
| `job_id` | Unique identifier |
| `status` | Current state (pending, running, completed, failed, cancelled) |
| `input_type` | Type of input (text, image, text_image) |
| `progress_percentage` | Overall progress 0-100 |
| `current_stage` | Current stage number (1-8) |
| `current_stage_name` | Human-readable stage name |
| `started_at` | When execution started |
| `created_at` | When job was created |
| `retry_count` | Number of retries performed |

---

## Timeline Example

```
Time    Event                               Status      Stage    Progress
────────────────────────────────────────────────────────────────────────
12:00:00  Job created                       pending     -        0%
12:00:00  Job enqueued                      pending     -        0%
12:00:01  Execution started                 running     1        10%
12:00:01  Stage 1 complete                  running     2        20%
12:00:01  Stage 2 complete                  running     3        35%
12:00:02  Stage 3 complete                  running     4        50%
12:00:02  Stage 4 complete                  running     5        70%
12:00:03  Stage 5 complete                  running     6        85%
12:00:04  Stage 6 complete                  running     7        95%
12:00:07  Stage 7 complete                  running     8        100%
12:00:08  Pipeline finished                 completed   8        100%
12:00:08  Result persisted                  completed   8        100%
```

---

## Event Sequence Diagram

### Success Path

```
Client                Server              Queue               Database
  │                     │                   │                    │
  ├─POST /analyze──────→│                   │                    │
  │                     ├─create Job───────────────────────────→│
  │ ← job_id           │                   │                    │
  │                     ├─enqueue──────────→│                    │
  │                     │                   ├─execute Pipeline   │
  │                     │                   │  (stages 1-8)      │
  │─GET /job/{id}──────→│  (repeated polling)                   │
  │ ← progress         │                   │                    │
  │                     │      ← progress events to WS           │
  │                     │                   ├─save Result────────→│
  │                     │                   │                    │
  │─GET /result/{id}──→│                   │                    │
  │ ← final result     │ (once completed)   │                    │
```

### Failure Path

```
Client                Server              Queue               Database
  │                     │                   │                    │
  ├─POST /analyze──────→│                   │                    │
  │ ← job_id           ├─create Job───────────────────────────→│
  │                     ├─enqueue──────────→│                    │
  │                     │                   ├─execute (fail)──→ │
  │                     │                   │  (retries, exits)  │
  │                     │      ← error event to WS               │
  │                     │                   │                    │
  │─GET /job/{id}──────→│                   │                    │
  │ ← status: failed   │ ← update status    │                    │
```

---

## Timeout Behavior

### Default Timeout
**Job Timeout:** 300 seconds (5 minutes)

### On Timeout
- Job marked as FAILED
- Error message: "Job execution exceeded timeout"
- WebSocket error event emitted
- Client informed via status endpoint

---

## Job Cleanup

### Automatic Cleanup
Jobs older than 7 days (configurable) with status in `[completed, failed, cancelled]` are automatically deleted.

```python
# config.py
JOB_CLEANUP_DAYS = 7
```

### Manual Cleanup
Use DELETE endpoint during pipeline execution to cancel a job:

```
DELETE /api/job/{job_id}
```

Completed/failed jobs cannot be canceled (use appropriate cleanup mechanism).

---

## Race Conditions & Edge Cases

### Late Subscription to WebSocket
If a client subscribes to a job that's already mid-execution:
1. WebSocket accepts connection
2. Latest cached event immediately sent
3. Client receives all subsequent events

### Multiple Concurrent Operations
- Two POST requests for same input: Creates 2 separate jobs ✓
- GET and DELETE simultaneously: DELETE wins, job marked cancelled ✓
- WebSocket subscribe after completion: Latest cached result sent ✓

### Database Consistency
- All job updates atomic (ACID compliance)
- Result persisted before completion status
- Stage records committed before next stage
