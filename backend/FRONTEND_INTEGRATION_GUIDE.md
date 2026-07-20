# Frontend Integration Guide

This guide shows how to integrate the HALLUCINATION_GUARD backend with your frontend application.

---

## Quick Start

### 1. Basic Job Submission

```javascript
async function submitAnalysis(inputText) {
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input_text: inputText,
      user_id: 'current-user-id'
    })
  });

  const { job_id, status, created_at } = await response.json();
  console.log(`Job created: ${job_id}, Status: ${status}`);
  return job_id;
}

// Usage
const jobId = await submitAnalysis('Is Paris the capital of Germany?');
```

### 2. Subscribe to Real-Time Progress

```javascript
function subscribeToProgress(jobId, onMessage, onError) {
  const ws = new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`);

  ws.onopen = () => {
    console.log('Connected to progress stream');
  };

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    onMessage(message);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError(error);
  };

  ws.onclose = () => {
    console.log('Disconnected from progress stream');
  };

  return ws;
}

// Usage
subscribeToProgress(
  jobId,
  (message) => {
    console.log(`Stage: ${message.data.name}, Progress: ${message.data.progress_percentage}%`);
  },
  (error) => console.error('Failed:', error)
);
```

### 3. Fetch Final Results

```javascript
async function getResults(jobId) {
  const response = await fetch(`http://localhost:8000/api/result/${jobId}`);
  
  if (!response.ok) {
    throw new Error('Results not yet available');
  }

  return await response.json();
}

// Usage
const results = await getResults(jobId);
console.log(`Hallucination: ${results.is_hallucination}`);
console.log(`Confidence: ${results.confidence}`);
console.log(`Explanation: ${results.explanation_text}`);
```

---

## Complete Example: React Component

```jsx
import React, { useState, useEffect, useRef } from 'react';

function HallucinationDetector() {
  const [inputText, setInputText] = useState('');
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  // Submit analysis
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setStatus('submitting');

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_text: inputText,
          user_id: 'user-123'
        })
      });

      if (!response.ok) throw new Error('Failed to submit analysis');

      const data = await response.json();
      setJobId(data.job_id);
      setStatus('analyzing');
      subscribeToProgress(data.job_id);
    } catch (err) {
      setError(err.message);
      setStatus('idle');
    }
  };

  // Subscribe to WebSocket progress
  const subscribeToProgress = (id) => {
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/progress/${id}`);

    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.message_type === 'stage_progress') {
        setProgress(message.data.progress_percentage);
      } else if (message.message_type === 'result') {
        setResults(message.data);
        setStatus('completed');
      } else if (message.message_type === 'error') {
        setError(message.data.error_message);
        setStatus('failed');
      }
    };

    wsRef.current.onerror = () => {
      setError('Lost connection to server');
    };

    wsRef.current.onclose = () => {
      if (status === 'analyzing') {
        setStatus('disconnected');
      }
    };
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Render
  return (
    <div className="detector-container">
      <h1>Hallucination Detector</h1>

      {status === 'idle' && (
        <form onSubmit={handleSubmit}>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to analyze..."
            rows={4}
          />
          <button type="submit" disabled={!inputText}>
            Analyze
          </button>
        </form>
      )}

      {status === 'analyzing' && (
        <div className="progress-container">
          <p>Analyzing... {progress}% complete</p>
          <progress value={progress} max={100} />
          <p>Job ID: {jobId}</p>
        </div>
      )}

      {status === 'completed' && results && (
        <div className="results-container">
          <h2>Analysis Complete</h2>
          <p>
            <strong>Hallucination Detected:</strong>{' '}
            {results.hallucination ? 'Yes' : 'No'}
          </p>
          <p>
            <strong>Confidence:</strong> {(results.confidence * 100).toFixed(1)}%
          </p>
          <p>
            <strong>Generated:</strong> {results.generated_response}
          </p>
          <p>
            <strong>Verified:</strong> {results.verified_answer}
          </p>
          <p>
            <strong>Explanation:</strong> {results.explanation}
          </p>
          <button onClick={() => window.location.reload()}>New Analysis</button>
        </div>
      )}

      {status === 'failed' && (
        <div className="error-container">
          <p className="error">{error}</p>
          <button onClick={() => window.location.reload()}>Try Again</button>
        </div>
      )}
    </div>
  );
}

export default HallucinationDetector;
```

---

## API Integration Patterns

### Pattern 1: Polling (No WebSocket)

For simple applications without WebSocket support:

```javascript
async function analyzeWithPolling(inputText) {
  // Submit job
  const jobResponse = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_text: inputText })
  });
  const { job_id } = await jobResponse.json();

  // Poll for completion
  let status = 'pending';
  while (status !== 'completed' && status !== 'failed') {
    await new Promise(r => setTimeout(r, 1000)); // wait 1s

    const statusResponse = await fetch(`http://localhost:8000/api/job/${job_id}`);
    const jobStatus = await statusResponse.json();
    status = jobStatus.status;

    console.log(`Progress: ${jobStatus.progress_percentage}%`);
  }

  // Get results
  const resultResponse = await fetch(`http://localhost:8000/api/result/${job_id}`);
  return await resultResponse.json();
}
```

### Pattern 2: WebSocket Only

For maximum efficiency with only WebSocket:

```javascript
function analyzeWithWebSocket(inputText) {
  return new Promise((resolve, reject) => {
    // Submit job
    fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input_text: inputText })
    })
      .then(r => r.json())
      .then(({ job_id }) => {
        // Subscribe to WebSocket
        const ws = new WebSocket(`ws://localhost:8000/ws/progress/${job_id}`);

        ws.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.message_type === 'result') {
            resolve(message.data);
            ws.close();
          }
        };

        ws.onerror = reject;
      })
      .catch(reject);
  });
}
```

### Pattern 3: Hybrid (WebSocket + Polling Fallback)

For reliability with fallback:

```javascript
function analyzeHybrid(inputText, options = {}) {
  const maxPollingRetries = options.maxPollingRetries || 60;
  let jobId;

  // Submit job
  return fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_text: inputText })
  })
    .then(r => r.json())
    .then(({ job_id }) => {
      jobId = job_id;

      // Try WebSocket
      return new Promise((resolve, reject) => {
        const ws = new WebSocket(`ws://localhost:8000/ws/progress/${job_id}`);
        let wsConnected = false;

        const wsTimeout = setTimeout(() => {
          if (!wsConnected) {
            console.warn('WebSocket timeout, falling back to polling');
            ws.close();
            fallbackToPoll(jobId, resolve, reject);
          }
        }, 5000);

        ws.onopen = () => {
          wsConnected = true;
          clearTimeout(wsTimeout);
        };

        ws.onmessage = (event) => {
          const message = JSON.parse(event.data);
          if (message.message_type === 'result') {
            resolve(message.data);
            ws.close();
          }
        };

        ws.onerror = () => {
          console.warn('WebSocket error, falling back to polling');
          fallbackToPoll(jobId, resolve, reject);
        };
      });
    });

  function fallbackToPoll(id, resolve, reject) {
    let attempts = 0;
    const poll = () => {
      fetch(`http://localhost:8000/api/job/${id}`)
        .then(r => r.json())
        .then(jobStatus => {
          if (jobStatus.status === 'completed' || jobStatus.status === 'failed') {
            return fetch(`http://localhost:8000/api/result/${id}`);
          }
          if (++attempts > maxPollingRetries) {
            throw new Error('Job timeout');
          }
          return new Promise(r => setTimeout(() => r(poll()), 1000));
        })
        .then(r => r.json())
        .then(resolve)
        .catch(reject);
    };
    poll();
  }
}
```

---

## Error Handling

```javascript
async function analyzeWithErrorHandling(inputText) {
  try {
    // Validation
    if (!inputText || inputText.trim().length === 0) {
      throw new Error('Input cannot be empty');
    }

    // Submission
    const response = await fetch('http://localhost:8000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input_text: inputText })
    });

    if (response.status === 400) {
      const data = await response.json();
      throw new Error(`Bad request: ${data.error}`);
    }
    if (response.status === 429) {
      throw new Error('Rate limited. Please wait before retrying.');
    }
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const { job_id } = await response.json();

    // Subscribe and process
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`ws://localhost:8000/ws/progress/${job_id}`);

      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket timeout'));
      }, 600000); // 10 minutes

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.message_type === 'error') {
          clearTimeout(timeout);
          reject(new Error(`Analysis failed: ${message.data.error_message}`));
        } else if (message.message_type === 'result') {
          clearTimeout(timeout);
          resolve(message.data);
        }
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('WebSocket connection error'));
      };
    });
  } catch (error) {
    console.error('Analysis error:', error);
    throw error;
  }
}
```

---

## UI Component Examples

### Progress Bar

```jsx
function ProgressBar({ percentage, stage, stageName }) {
  return (
    <div className="progress-bar-container">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${percentage}%` }} />
      </div>
      <p>
        Stage {stage}: {stageName}
      </p>
      <p>{percentage}% Complete</p>
    </div>
  );
}
```

### Results Display

```jsx
function ResultsDisplay({ results }) {
  return (
    <div className="results">
      <div className={`verdict ${results.hallucination ? 'hallucination' : 'truthful'}`}>
        {results.hallucination ? '⚠️ Hallucination Detected' : '✓ Appears Truthful'}
      </div>

      <div className="confidence">
        <label>Confidence:</label>
        <div className="confidence-bar">
          <div 
            className="confidence-fill" 
            style={{ width: `${results.confidence * 100}%` }} 
          />
        </div>
        <span>{(results.confidence * 100).toFixed(1)}%</span>
      </div>

      <div className="explanation">
        <h3>Explanation</h3>
        <p>{results.explanation}</p>
      </div>

      <div className="evidence">
        <h3>Retrieved Evidence</h3>
        <ul>
          {results.retrieved_evidence?.supporting_documents?.map((doc, i) => (
            <li key={i}>{doc}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

---

## Testing

```javascript
// Mock WebSocket for testing
class MockWebSocket {
  constructor(url) {
    this.url = url;
    setTimeout(() => {
      this.onopen?.();
      // Simulate stage progress
      setTimeout(() => {
        this.onmessage?.({
          data: JSON.stringify({
            message_type: 'stage_progress',
            data: { progress_percentage: 50 }
          })
        });
      }, 100);
      // Simulate completion
      setTimeout(() => {
        this.onmessage?.({
          data: JSON.stringify({
            message_type: 'result',
            data: { hallucination: true, confidence: 0.95 }
          })
        });
      }, 200);
    }, 0);
  }
  close() {}
}

// In test file
global.WebSocket = MockWebSocket;
```

---

## CORS Configuration

If frontend is on different origin, ensure CORS is enabled on backend:

```python
# config.py
CORS_ORIGINS = [
  "http://localhost:3000",  # React dev server
  "http://localhost:5173",  # Vite dev server
  "https://yourdomain.com"  # Production
]
```

---

## Rate Limiting

Backend limits API to 60 requests/minute per IP:

```javascript
async function analyzeWithRetry(inputText, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_text: inputText })
      });

      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        const waitMs = (parseInt(retryAfter) || (attempt + 1)) * 1000;
        await new Promise(r => setTimeout(r, waitMs));
        continue;
      }

      return await response.json();
    } catch(error) {
      if (attempt === maxRetries - 1) throw error;
    }
  }
}
```

---

## Resources

- **WebSocket Doc:** [WEBSOCKET_DOCUMENTATION.md](./WEBSOCKET_DOCUMENTATION.md)
- **API Doc:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- **Architecture:** [PIPELINE_ARCHITECTURE.md](./PIPELINE_ARCHITECTURE.md)
- **Sample Payloads:** [tests/sample_event_payloads.json](./tests/sample_event_payloads.json)
