# Progress Bar & Timeline UI Specifications

## Progress Bar Component

### Visual Design

```
┌──────────────────────────────────────────────────┐
│ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                  │
│ 35% Complete | Elapsed: 3.2s | Est: 5.8s      │
└──────────────────────────────────────────────────┘
```

### Dimensions
- **Width:** 100% of container (min 300px, max 800px)
- **Height:** 24px (bar) + 20px (labels)
- **Container Padding:** 16px

### Colors

#### Bar Gradient by Progress
```
0%     ← Blue (#0066CC)
25%    ← Cyan (#00CCCC)
50%    ← Green (#00CC00)
75%    ← Lime (#CCFF00)
100%   ← Lime (#CCFF00)
```

#### Alternative: Solid Colors (adjust as needed)
```
0-25%:  #0066CC (Blue)
25-50%: #00CCCC (Cyan)
50-75%: #00CC00 (Green)
75-100%: #CCFF00 (Yellow/Lime)
```

### CSS Implementation

```css
.progress-bar {
  width: 100%;
  height: 24px;
  background-color: #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #0066CC, #CCFF00);
  width: 35%;
  transition: width 0.3s ease-out;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
}

.progress-text {
  color: white;
  font-weight: bold;
  font-size: 12px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.progress-labels {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #666;
  margin-top: 4px;
}
```

### Data Binding

```typescript
interface ProgressData {
  percentage: number;        // 0-100
  elapsedSeconds: number;   // calculated from start_time
  estimatedSeconds: number; // extrapolated
  currentStage: number;     // 1-8
  stageName: string;        // displayed in label
  status: 'pending' | 'running' | 'completed';
}

function updateProgressBar(data: ProgressData) {
  // Update bar width
  barFill.style.width = `${data.percentage}%`;
  
  // Update percentage text
  percentageText.innerText = `${Math.round(data.percentage)}%`;
  
  // Update time labels
  elapsedLabel.innerText = formatSeconds(data.elapsedSeconds);
  estimatedLabel.innerText = formatSeconds(data.estimatedSeconds);
  
  // Update stage display
  stageLabel.innerText = `Stage ${data.currentStage}/8: ${data.stageName}`;
}
```

---

## Stage Timeline Component

### Visual Design

```
Compact Timeline (mobile)
┌─────────────────────────────────┐
│ ●─●─●─●─●─●─●─●                │
│ 1 2 3 4 5 6 7 8                 │
│ ✓ ✓ ◐ ◯ ◯ ◯ ◯ ◯                │
│                                 │
│ Stage 3: Hidden State Extract...│
│ 0.8s                            │
└─────────────────────────────────┘

Expanded Timeline (desktop)
┌─────────────────────────────────────────────────┐
│ ●─────●─────●─────●─────●─────●─────●─────●    │
│ 1     2     3     4     5     6     7     8      │
│ ✓     ✓     ◐     ◯     ◯     ◯     ◯     ◯     │
│ 0.5s  0.6s  0.8s  ...                           │
│                                                  │
│ Current: Stage 3 - Hidden State Extraction      │
│ Status: Running (1200ms elapsed)                │
└─────────────────────────────────────────────────┘
```

### Stage Indicators

| Symbol | Meaning | Color |
|--------|---------|-------|
| ✓ | Completed | Green (#4CAF50) |
| ◐ | In Progress | Blue (#0066CC) |
| ◯ | Pending | Gray (#CCCCCC) |
| ✗ | Failed | Red (#FF6B6B) |

### CSS Implementation

```css
.timeline {
  display: flex;
  align-items: flex-start;
  gap: 4px;
  margin: 16px 0;
}

.timeline-connector {
  flex: 1;
  height: 2px;
  background-color: #ddd;
  margin-top: 8px;
}

.timeline-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.stage-dot {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  color: white;
  border: 2px solid #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.stage-dot.completed {
  background-color: #4CAF50;
}

.stage-dot.running {
  background-color: #0066CC;
  animation: pulse 1.5s infinite;
}

.stage-dot.pending {
  background-color: #CCCCCC;
}

.stage-dot.failed {
  background-color: #FF6B6B;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
  50% {
    box-shadow: 0 2px 8px rgba(0, 102, 204, 0.6);
  }
}

.stage-number {
  font-size: 11px;
  color: #666;
  font-weight: bold;
  margin-top: 4px;
}

.stage-duration {
  font-size: 10px;
  color: #999;
  min-width: 40px;
  text-align: center;
}
```

### React Component Example

```jsx
function StageTimeline({ stages, currentStage }) {
  return (
    <div className="timeline-container">
      <div className="timeline">
        {stages.map((stage, index) => (
          <React.Fragment key={stage.id}>
            <div className="timeline-stage">
              <div className={`stage-dot ${stage.status}`}>
                {stage.status === 'completed' && '✓'}
                {stage.status === 'running' && '◐'}
                {stage.status === 'failed' && '✗'}
              </div>
              <div className="stage-number">{index + 1}</div>
              {stage.duration_ms && (
                <div className="stage-duration">
                  {(stage.duration_ms / 1000).toFixed(1)}s
                </div>
              )}
            </div>
            {index < stages.length - 1 && <div className="timeline-connector" />}
          </React.Fragment>
        ))}
      </div>

      <div className="timeline-info">
        <p className="stage-title">
          Stage {currentStage}: {stages[currentStage - 1]?.name}
        </p>
        <p className="stage-status">
          Status: {stages[currentStage - 1]?.status}
        </p>
      </div>
    </div>
  );
}
```

---

## Timeline (Sequence Diagram)

### Format: Vertical Timeline

```
┌────────────────────────────────┐
│ 12:00:00 | Job Created         │
│          └─ PENDING            │
├────────────────────────────────┤
│ 12:00:01 | Stage 1 Complete    │
│          └─ RUNNING (10%)      │
├────────────────────────────────┤
│ 12:00:01 | Stage 2 Complete    │
│          └─ RUNNING (20%)      │
├────────────────────────────────┤
│ 12:00:02 | Stage 3 Running     │
│          └─ RUNNING (35%)      │
│          └─ ⏱️ 0.8s elapsed   │
├────────────────────────────────┤
│ 12:00:08 | All Stages Done     │
│          └─ COMPLETED (100%)   │
├────────────────────────────────┤
│ 12:00:08 | Results Available   │
│          └─ Ready for display  │
└────────────────────────────────┘
```

### Implementation

```jsx
function ExecutionTimeline({ events }) {
  return (
    <div className="timeline-list">
      {events.map((event, index) => (
        <div key={index} className={`timeline-entry ${event.type}`}>
          <div className="timeline-time">
            {new Date(event.timestamp).toLocaleTimeString()}
          </div>
          <div className="timeline-icon">
            {event.type === 'stage_progress' && '▶'}
            {event.type === 'result' && '✓'}
            {event.type === 'error' && '✗'}
          </div>
          <div className="timeline-content">
            <p className="timeline-title">{event.title}</p>
            <p className="timeline-detail">{event.detail}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
```

---

## Progress Indicator States

### Pending State
```
Color: #CCCCCC (Light Gray)
Animation: None
Icon: ◯
Opacity: 0.5
```

### Running State
```
Color: #0066CC (Blue)
Animation: Pulse (1.5s)
Icon: ◐
Opacity: 1.0
```

### Completed State
```
Color: #4CAF50 (Green)
Animation: None
Icon: ✓
Opacity: 1.0
```

### Failed State
```
Color: #FF6B6B (Red)
Animation: Shake (500ms)
Icon: ✗
Opacity: 1.0
```

---

## Responsive Behavior

### Mobile (< 768px)
- Timeline: Horizontal, compact (no labels)
- Progress: Full width, simplified
- Time display: Clock time only
- Stage duration: Inline text

### Tablet (768px - 1024px)
- Timeline: Horizontal with stage numbers
- Progress: 80% width, centered
- Time display: Both elapsed and estimated
- Stage duration: Below dot

### Desktop (> 1024px)
- Timeline: Expanded with all details
- Progress: Full width with gradient
- Time display: Detailed breakdown
- Stage duration: Formatted (X.XXs)

---

## Interaction Patterns

### Click on Stage Dot
```javascript
stageElement.addEventListener('click', (e) => {
  const stageNumber = e.target.dataset.stage;
  showStageDetails(stageNumber);
});
```

### Hover Effects
```css
.stage-dot:hover {
  transform: scale(1.2);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  cursor: pointer;
}
```

### Stage Details Popup
```jsx
<div className="stage-details-popup">
  <div className="popup-header">Stage 3: Hidden State Extraction</div>
  <div className="popup-content">
    <div className="detail-row">
      <span className="label">Status:</span>
      <span className="value">Completed</span>
    </div>
    <div className="detail-row">
      <span className="label">Duration:</span>
      <span className="value">850ms</span>
    </div>
    <div className="detail-row">
      <span className="label">Started:</span>
      <span className="value">12:00:01.200Z</span>
    </div>
  </div>
</div>
```

---

## Animation Specifications

### Progress Bar Fill Animation
```css
@keyframes fillProgress {
  from {
    width: var(--old-width, 0%);
  }
  to {
    width: var(--new-width, 100%);
  }
}

.progress-fill {
  animation: fillProgress 0.3s ease-out;
}
```

### Stage Pulse Animation
```css
@keyframes stagePulse {
  0% {
    box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
  }
  50% {
    box-shadow: 0 2px 8px rgba(0, 102, 204, 0.6);
  }
  100% {
    box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
  }
}

.stage-dot.running {
  animation: stagePulse 1.5s infinite;
}
```

### Shake on Error
```css
@keyframes shakeError {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-4px); }
  75% { transform: translateX(4px); }
}

.stage-dot.failed {
  animation: shakeError 0.5s;
}
```

---

## Event Data Model

```typescript
interface StageEvent {
  job_id: string;
  stage: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_percentage: number;
  start_time: string;      // ISO 8601
  end_time?: string;       // ISO 8601
  duration_ms?: number;
  timestamp: string;       // ISO 8601
}

interface ProgressState {
  jobId: string;
  stages: StageEvent[];
  currentStageIndex: number;
  overallProgress: number;
  startTime: Date;
  elapsedMs: number;
  estimatedTotalMs: number;
}

// Calculate elapsed time
function getElapsedTime(startTime: Date): number {
  return Date.now() - startTime.getTime();
}

// Extrapolate remaining time
function estimateRemainingTime(
  currentProgress: number,
  elapsedMs: number
): number {
  if (currentProgress === 0) return 0;
  const totalMs = (elapsedMs / currentProgress) * 100;
  return totalMs - elapsedMs;
}
```

---

## Accessibility

### ARIA Labels
```html
<div
  className="progress-bar"
  role="progressbar"
  aria-valuenow="35"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-label="Job progress: 35% complete"
>
  <div className="progress-fill" />
</div>
```

### Semantic HTML
```html
<section aria-label="Execution Timeline">
  <ol className="timeline">
    <li className="stage completed" aria-label="Stage 1: Input Received, completed">
      <!-- content -->
    </li>
  </ol>
</section>
```

### Keyboard Navigation
```javascript
// Stage dots should be tab-focusable
stageElement.setAttribute('tabindex', '0');

// Handle Enter/Space
stageElement.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    showStageDetails();
  }
});
```
