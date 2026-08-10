# Hallucination-Detection Backend — All 12 Phases

A FastAPI backend that takes a prompt, generates a response (Ollama),
breaks it into atomic factual claims, extracts entities, retrieves
evidence (Wikipedia + optional FEVER), verifies each claim, and produces
a hallucination score with a full explanation — persisted to SQLite.

## Phase map (where each phase lives)

| Phase | What | Where |
|---|---|---|
| 1 | Project init, config, logging, DB, health, jobs, UUIDs | `main.py`, `config/`, `core/`, `database/base.py`, `api/routes/health.py`, `api/routes/jobs.py` |
| 2 | Database layer (Result, PipelineStage, Claim tables + repos) | `database/models.py`, `services/result_service.py`, `services/stage_service.py` |
| 3 | LLM service (Ollama, generation only) | `services/llm_service.py` |
| 4 | Pipeline engine (orchestration, timing, parallel retrieval) | `execution/manager.py` |
| 4b | Query grounding (pre-generation retrieval on the query itself) | `pipeline/stages/query_grounding.py` |
| 5 | Claim extraction (LLM-assisted + rule-based fallback) | `pipeline/stages/claim_extraction.py` |
| 6 | Entity extraction / NER (spaCy + rule-based fallback) | `knowledge_base/ner.py`, `pipeline/stages/entity_extraction.py` |
| 7 | Retrieval (Wikipedia, FEVER, ranking, caching) | `retrieval/*.py`, `pipeline/stages/retrieval_*.py`, `evidence_ranking.py` |
| 8 | Verification (LLM-assisted + lexical fallback) | `hallucination/verification.py`, `pipeline/stages/verification.py` |
| 8b | Query consistency (denial-despite-evidence + fabricated-alternative detection) | `pipeline/stages/query_consistency.py` |
| 9 | Hallucination detection (claims-only, never the prompt) | `hallucination/detector.py`, `pipeline/stages/hallucination_detection.py` |
| 10 | Explainability (verified answer, evidence, contradictions) | `hallucination/explainability.py`, `pipeline/stages/explainability.py` |
| 11 | Public API (`/api/analyze`, `/api/job`, `/api/result`, `/api/history`) | `api/routes/analyze.py` |
| 12 | Optimization (cache, parallel retrieval, retry, timing logs) | `retrieval/cache.py`, `utils/retry.py`, `execution/manager.py`, `main.py` middleware |

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # already provided; edit if needed
```

Requires **Python 3.10+** (the code uses `X | None` union syntax throughout).

### Getting full quality out of it (optional but recommended)

The backend runs with just `requirements.txt` installed, using rule-based
fallbacks everywhere an ML model would normally help. To get the real thing:

```bash
# Better NER (Phase 6) - without this, falls back to a conservative
# gazetteer + regex recognizer
pip install spacy
python -m spacy download en_core_web_sm

# Better evidence ranking (Phase 7) - without this, falls back to
# lexical word-overlap scoring
pip install sentence-transformers faiss-cpu

# LLM generation, claim extraction, and verification (Phases 3, 5, 8)
# all need Ollama running locally:
ollama pull llama3.2:3b
ollama serve
```

FEVER retrieval (Phase 7) needs a local copy of the FEVER dataset — set
`FEVER_DATASET_PATH` in `.env` to a JSONL file of
`{"claim", "label", "evidence_text", "wiki_page"}` rows. Without it,
FEVER retrieval just contributes no evidence (logged once, not an error).

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Try it end to end

```bash
# 1. Kick off analysis (returns immediately, runs in the background)
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Lamine Yamal and where he plays football"}'
# -> {"job_id": "...", "status": "pending"}

# 2. Poll job status
curl http://localhost:8000/api/job/<job_id>

# 3. See per-stage timing once it's running/done
curl http://localhost:8000/api/job/<job_id>/stages

# 4. Get the full result once status is "completed"
curl http://localhost:8000/api/result/<job_id>

# 5. See past runs
curl http://localhost:8000/api/history

# Basic liveness + LLM reachability
curl http://localhost:8000/health
curl http://localhost:8000/health/llm
```

`GET /api/result/<job_id>` returns:
```json
{
  "job_id": "...",
  "generated_response": "...",
  "verified_answer": "...",
  "explanation": "3 claim(s) checked: 2 supported, 0 contradicted, 1 insufficient evidence.",
  "overall_confidence": 0.71,
  "hallucination_score": 0.17,
  "processing_time_ms": 4213.5,
  "claims": [
    {"text": "...", "entities": [...], "verdict": "supported", "confidence": 0.82, "evidence": [...]}
  ],
  "created_at": "..."
}
```

Interactive docs: http://localhost:8000/docs

## Design notes / things to know before you extend this

- **Claim extraction and verification are LLM-assisted with automatic
  fallback.** If Ollama is unreachable or returns malformed output, both
  stages fall back to rule-based/lexical methods rather than failing the
  job — check `pipeline_stages.error_message` and the logs to see which
  path was actually used for a given run.
- **Retrieval priority** (`retrieval/ranker.py`) always resolves PERSON
  entities before LOCATION/COUNTRY ones — this is the fix for the
  "searches Africa before Lamine Yamal" bug described in the spec.
- **Hallucination scoring never sees the user's original prompt** — only
  claim text and verdicts (`hallucination/detector.py`), by construction:
  `PipelineContext.query` simply isn't passed into that module. Query-
  awareness (see below) is handled entirely in the stages *around* it, which
  turn query-level findings into ordinary claims before scoring ever runs.
- **Query grounding runs before generation** (`pipeline/stages/query_grounding.py`):
  it extracts entities from the query itself and retrieves Wikipedia
  evidence for them, then feeds that as reference material into the LLM's
  system prompt (`generation.py`). This is the fix for the model
  confidently denying a well-known entity exists at all — it's given the
  evidence up front instead of only being checked after the fact.
- **Query consistency runs after verification, before hallucination
  detection** (`pipeline/stages/query_consistency.py`): if the response
  denies knowing about an entity the query-grounding evidence confirms is
  real, that's added as its own CONTRADICTED claim. If the response shows
  the "couldn't find X ... however/but I found Y" fabricated-alternative
  shape, claims about Y (anything not from the query itself) get flagged
  via `ClaimContext.fabricated_alternative` and are weighed close to
  CONTRADICTED in the score, even though verification could only mark them
  INSUFFICIENT (there's rarely a source that explicitly refutes a made-up
  name — the fabrication *pattern* is itself the signal).
- **Wikipedia + FEVER retrieval run concurrently** inside
  `execution/manager.py`, but their DB timing writes (start/finish) are
  kept sequential — a single SQLAlchemy `AsyncSession` isn't safe to use
  from two coroutines simultaneously, so only the actual HTTP calls are
  parallelized.
- **Non-critical stages degrade instead of failing the job.** Each
  `Stage` has a `critical` flag (`pipeline/stages/base.py`); retrieval
  and explainability are non-critical, generation/verification/
  hallucination-detection are critical.
- **`/api/jobs`** (from Phase 1) still exists as a plain job-creation
  endpoint with no pipeline attached — useful for testing the DB layer
  in isolation. For actual analysis, use **`/api/analyze`**.

## What I could and couldn't verify here

Built and syntax-checked (`python -m py_compile` on all 36 modules) in a
sandbox with no network access, so I could not: `pip install` anything,
run `ollama`, hit the live Wikipedia API, or boot uvicorn end-to-end here.
Everything above is real, complete code — please run the steps above
locally and let me know what breaks so I can fix it directly rather than
guessing.
