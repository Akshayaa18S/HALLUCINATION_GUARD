# Hallucination Guard — Frontend

A Vite + React + TypeScript + Tailwind dashboard implementing the "New Analysis"
screen: prompt input, live 8-stage execution pipeline, generated vs. verified
answer, ensemble confidence breakdown, retrieved evidence, and request details.

## Getting Started

```bash
cd frontend
npm install
cp .env.example .env.local   # point at your backend, if not localhost:8000
npm run dev
```

The app talks to the backend at `VITE_API_BASE_URL` (`POST /api/analyze`,
`WS /ws/progress/{job_id}`). If no backend is reachable, it automatically
falls back to a local simulated run so the UI is still fully explorable.

## Structure

```
frontend/
├── src/
│   ├── components/       # Sidebar, panels, pipeline, result tabs, gauges
│   ├── api.ts             # REST submit + WebSocket streaming (+ offline demo fallback)
│   ├── types.ts           # Shared types + the 8 pipeline stage definitions
│   ├── App.tsx             # Layout + state orchestration
│   └── index.css
├── tailwind.config.js     # Dark dashboard color tokens
└── vite.config.ts
```

## Frontend Technology Stack

- **Framework**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Icons**: lucide-react
- **HTTP/WebSocket**: native `fetch` + `WebSocket` API

## Original Planning Notes

The structure and phased feature list below were the original plan for this
directory; Phase 1–3 (input, live pipeline, results display) are now built.

## Planned Structure

```
frontend/
├── public/                          # Static assets
├── src/
│   ├── components/                  # Reusable React components
│   │   ├── common/                  # Common UI components
│   │   ├── pipeline/                # Pipeline visualization
│   │   │   ├── StageDisplay.tsx    # Individual stage display
│   │   │   ├── ProgressTimeline.tsx # Timeline of stages
│   │   │   └── HeatmapViewer.tsx   # Attention heatmap viewer
│   │   └── results/                 # Result display components
│   ├── pages/                       # Next.js pages or route pages
│   │   ├── index.tsx               # Home / Input form
│   │   ├── analysis/               # Analysis in progress
│   │   └── results/                # Results display
│   ├── hooks/                       # Custom React hooks
│   │   ├── useWebSocket.ts         # WebSocket hook
│   │   ├── useJobStatus.ts         # Job status polling
│   │   └── usePipeline.ts          # Pipeline state management
│   ├── services/                    # API client services
│   │   ├── api.ts                  # REST API client
│   │   ├── websocket.ts            # WebSocket service
│   │   └── jobService.ts           # Job-related API calls
│   ├── store/                       # Redux or Zustand store
│   │   ├── slices/
│   │   │   ├── jobSlice.ts
│   │   │   ├── pipelineSlice.ts
│   │   │   └── resultsSlice.ts
│   │   └── store.ts
│   ├── types/                       # TypeScript type definitions
│   │   ├── job.ts
│   │   ├── pipeline.ts
│   │   └── api.ts
│   ├── utils/                       # Utility functions
│   │   ├── formatters.ts
│   │   └── validators.ts
│   ├── styles/                      # Global styles
│   ├── App.tsx
│   └── index.tsx
├── tests/                           # Test files
├── .env.example                     # Environment template
├── .gitignore
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── README.md
```

## Features to Implement

### Phase 1: Input & Analysis Submission
- [ ] Text input form
- [ ] Image upload
- [ ] Analysis submission
- [ ] Job ID generation

### Phase 2: Real-Time Pipeline Visualization
- [ ] WebSocket connection management
- [ ] Stage progress tracking
- [ ] Real-time UI updates
- [ ] Animated loading indicators
- [ ] Progress timeline display

### Phase 3: Results Display
- [ ] Hallucination score display
- [ ] Confidence percentage
- [ ] Generated vs verified answer
- [ ] Retrieved evidence display
- [ ] SHAP explanations
- [ ] Attention heatmaps

### Phase 4: Advanced Features
- [ ] History of previous analyses
- [ ] Export results as PDF/JSON
- [ ] Batch analysis
- [ ] Custom configurations
- [ ] User accounts (optional)

## Getting Started

Once frontend development begins:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

## API Integration

The frontend will connect to the backend at:
- **REST API**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:8000/ws/progress/{job_id}`

## Environment Configuration

Create `.env.local`:
```env
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

---

**Status**: ✅ Phase 1–3 implemented (input, live pipeline, results). Phase 4 (history, export, batch, accounts) still open.
