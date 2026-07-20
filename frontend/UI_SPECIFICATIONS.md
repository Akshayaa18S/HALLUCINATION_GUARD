# Frontend UI Specifications & Expected Event Formats

## UI Component Overview

The frontend should implement the following main components for consuming HALLUCINATION_GUARD events:

```
┌─────────────────────────────────────────────────┐
│           Analysis Input Form                   │
│  ┌───────────────────────────────────────────┐  │
│  │ Text Input (required)                     │  │
│  │ Image Upload (optional)                   │  │
│  │ [Submit] [Clear]                          │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           Progress Display                      │
│  ┌───────────────────────────────────────────┐  │
│  │ Current Stage: Generating Response (2/8) │  │
│  │ ████████░░░░░░░░░░ 35%                    │  │
│  │ Elapsed: 3.2s | Est. Remaining: 2.1s    │  │
│  │ [Cancel Job]                              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           Results Display                       │
│  ┌───────────────────────────────────────────┐  │
│  │ ⚠️  HALLUCINATION DETECTED                │  │
│  │ Confidence: ████████░░ 94.2%              │  │
│  │                                            │  │
│  │ Generated: "Paris is capital of Germany"  │  │
│  │ Verified:  "Paris is capital of France"   │  │
│  │ Explanation: The model ignored context... │  │
│  │ [New Analysis] [Share] [Download]         │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Input Component

### Layout
```
┌─────────────────────────────────────────┐
│ Text Analysis                           │
├─────────────────────────────────────────┤
│ Enter text to analyze (required):       │
│ ┌───────────────────────────────────┐  │
│ │ Is Paris the capital of Germany? │  │
│ │                                   │  │
│ │                                   │  │
│ └───────────────────────────────────┘  │
│ [+ Add Image] [Submit] [Clear]          │
├─────────────────────────────────────────┤
│ Optional: Upload image                  │
│ [Choose File] No file selected           │
└─────────────────────────────────────────┘
```

### State Management
- Text input: max 5000 characters
- Image: max 10MB, JPG/PNG
- Submit button: enabled only if text is present
- Clear button: resets all fields

### Validation
- Show error if text empty on submit
- Show error if image > 10MB
- Show warning if text > 4000 chars

---

## Progress Component

### Layout During Analysis
```
┌─────────────────────────────────────────┐
│ Job Progress                            │
├─────────────────────────────────────────┤
│ ID: a1b2c3d4-e5f6-4789-a123-456789...  │
│                                         │
│ Stage 3 of 8: Hidden State Extraction  │
│ ████████░░░░░░░░░░░░░░░░░░ 35%         │
│                                         │
│ Time: 00:03 | Est. Total: 00:08        │
│                                         │
│ [Cancel Job]                            │
└─────────────────────────────────────────┘
```

### Progress Bar Details
- Width fills container (0-100%)
- Color changes by progress:
  - 0-25%: Blue (#0066CC)
  - 25-50%: Cyan (#00CCCC)
  - 50-75%: Green (#00CC00)
  - 75-100%: Lime (#CCFF00)
- Shows percentage text inside bar

### Stage Display
- Show full stage name and number (e.g., "Stage 2 of 8")
- Update stage on `stage_progress` events with new stage number

### Time Display
- Elapsed: calculated from start_time
- Est. Remaining: extrapolated from current progress
- Format: MM:SS

### Cancel Button
- Calls DELETE `/api/job/{job_id}`
- Shows confirmation: "Cancel analysis?"
- Disables after click

---

## Results Component

### Layout - Hallucination Detected
```
┌─────────────────────────────────────────┐
│ Analysis Results                        │
├─────────────────────────────────────────┤
│ ⚠️  HALLUCINATION DETECTED              │
│                                         │
│ Confidence Score                        │
│ ███████░░░░░░░░░░░░░░░░░░░░░░ 94.2%   │
│                                         │
│ Original Response                       │
│ "Paris is the capital of Germany"      │
│                                         │
│ Verified Answer                         │
│ "Paris is the capital of France"        │
│                                         │
│ Explanation                             │
│ The model incorrectly claimed that...   │
│                                         │
│ Sources & Evidence                      │
│ • Wikipedia: France article             │
│ • FEVER: Claims about capitals          │
│                                         │
│ [New Analysis] [Share] [Report Issue]   │
└─────────────────────────────────────────┘
```

### Layout - Truthful
```
┌─────────────────────────────────────────┐
│ Analysis Results                        │
├─────────────────────────────────────────┤
│ ✓ APPEARS TRUTHFUL                      │
│                                         │
│ Confidence Score                        │
│ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12.1% │
│                                         │
│ Original Response                       │
│ "Paris is the capital of France"        │
│                                         │
│ Verified Answer                         │
│ "Paris is the capital of France"        │
│                                         │
│ Explanation                             │
│ The response matches verified sources.  │
│                                         │
│ Verification Status: Confirmed          │
│                                         │
│ [New Analysis] [Share]                  │
└─────────────────────────────────────────┘
```

### Result Fields

| Field | Display | Format |
|-------|---------|--------|
| `hallucination` | Verdict badge | "⚠️ HALLUCINATION" or "✓ TRUTHFUL" |
| `confidence` | Bar + percentage | Bar 0-100%, text e.g. "94.2%" |
| `generated_response` | Quote | Quoted text, wrapped at 80 chars |
| `verified_answer` | Quote | Quoted text, wrapped at 80 chars |
| `explanation` | Paragraph | Plain text, preserve line breaks |
| `retrieved_evidence.sources` | List | Comma-separated sources |
| `retrieved_evidence.supporting_documents` | Bullets | One per line with • prefix |
| `processing_time_ms` | Footer | "Processed in X.XXs" |

---

## Error States

### Network Error
```
┌─────────────────────────────────────────┐
│ ❌ Connection Failed                    │
│ Unable to connect to analysis server     │
│                                         │
│ [Retry] [Go Back]                       │
└─────────────────────────────────────────┘
```

### Analysis Failed
```
┌─────────────────────────────────────────┐
│ ❌ Analysis Failed                      │
│ Stage 3 failed: Unable to extract       │
│ hidden states after 3 attempts          │
│                                         │
│ [Try Again] [Report Issue]              │
└─────────────────────────────────────────┘
```

### Rate Limited
```
┌─────────────────────────────────────────┐
│ ⏳ Rate Limit Exceeded                 │
│ Too many requests. Wait 30 seconds.     │
│                                         │
│ [Retry in 30s] [Cancel]                 │
└─────────────────────────────────────────┘
```

---

## Stage Timeline

### Visual Timeline
```
┌─────────────────────────────────────────────────┐
│ ●─────●─────●─────●─────●─────●─────●─────●    │
│ 1     2     3     4     5     6     7     8      │
│ ✓     ✓     ◐     ◐     ◯     ◯     ◯     ◯     │
│ 0.5s  0.6s  0.8s  0.9s  ...                     │
└─────────────────────────────────────────────────┘

Legend:
  ✓ = Completed
  ◐ = In Progress
  ◯ = Pending
```

### Stage Names (in order)
1. Input Received
2. Generating Response
3. Hidden State Extraction
4. Feature Extraction
5. Hallucination Detection
6. Fact Verification
7. Explainability
8. Analysis Completed

---

## Expected Event Format Reference

### Event: Stage Progress
```json
{
  "message_type": "stage_progress",
  "data": {
    "job_id": "a1b2c3d4-e5f6-...",
    "stage": 3,
    "name": "Hidden State Extraction",
    "status": "running",
    "progress_percentage": 35.0,
    "start_time": "2026-07-15T12:00:02Z",
    "end_time": null,
    "duration_ms": null,
    "metadata": {
      "token_embeddings": [0.12, 0.34, 0.56],
      "attention_maps": [0.11, 0.22, 0.33]
    },
    "error_message": null
  },
  "timestamp": "2026-07-15T12:00:02.500Z"
}
```

**UI Action:**
- Update current stage display to "Stage 3 of 8"
- Update progress bar to 35%
- Update stage name to "Hidden State Extraction"
- Update elapsed time

### Event: Final Result
```json
{
  "message_type": "result",
  "data": {
    "job_id": "a1b2c3d4-e5f6-...",
    "status": "completed",
    "hallucination": true,
    "confidence": 0.942,
    "generated_response": "Paris is the capital of Germany.",
    "verified_answer": "Paris is the capital of France.",
    "retrieved_evidence": {
      "sources": ["Wikipedia"],
      "supporting_documents": ["France article"]
    },
    "explanation": "The response conflicts with retrieved evidence...",
    "processing_time_ms": 4200.0
  },
  "timestamp": "2026-07-15T12:00:08Z"
}
```

**UI Action:**
- Hide progress component
- Show results component
- Populate all result fields
- Display processing time: "Processed in 4.2s"

### Event: Error
```json
{
  "message_type": "error",
  "data": {
    "job_id": "a1b2c3d4-e5f6-...",
    "stage": 5,
    "error_message": "Failed to load hallucination detection model",
    "timestamp": "2026-07-15T12:00:05Z"
  },
  "timestamp": "2026-07-15T12:00:05Z"
}
```

**UI Action:**
- Hide progress component
- Show error component
- Display error message
- Show retry button

---

## Color Scheme

### Verdict Colors
- **Hallucination:** `#FF6B6B` (Red)
- **Truthful:** `#4CAF50` (Green)
- **Unknown:** `#FFC107` (Yellow)

### Progress Bar
- **0-25%:** `#0066CC` (Blue)
- **25-50%:** `#00CCCC` (Cyan)
- **50-75%:** `#00CC00` (Green)
- **75-100%:** `#CCFF00` (Yellow)

### Confidence Bar
- **Low (0-33%):** `#FF6B6B` (Red)
- **Medium (33-66%):** `#FFC107` (Yellow)
- **High (66-100%):** `#4CAF50` (Green)

---

## Typography

- **Heading 1:** 32px, bold, color: #333
- **Heading 2:** 24px, bold, color: #666
- **Body:** 16px, regular, color: #666
- **Label:** 14px, medium, color: #999
- **Code:** Monospace, 12px, `#F5F5F5` background

---

## Responsive Design

### Mobile (< 768px)
- Single column layout
- Smaller fonts (80% of desktop)
- Full-width input field
- Stacked buttons

### Tablet (768px - 1024px)
- Single column, centered
- Max width: 600px
- Standard fonts

### Desktop (> 1024px)
- Max width: 800px
- Centered with padding
- Side-by-side sections where possible

---

## Accessibility Requirements

- Alt text for icons (e.g., "⚠️ Warning", "✓ Success")
- ARIA labels for interactive elements
- Keyboard navigation support
- High contrast colors (WCAG AA)
- Focus indicators on buttons
- Semantic HTML structure

---

## Animation Guidelines

- Progress bar: Smooth linear transition
- Stage updates: Fade in/out (200ms)
- Result reveal: Slide up (300ms)
- Error display: Pulse highlight (500ms)
- Disable animations option in settings
