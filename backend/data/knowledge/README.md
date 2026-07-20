# Knowledge base seed data

These `.jsonl` files are the **static** portion of the retrieval corpus that
`services/knowledge_base.py` embeds and indexes once at startup. Each line is
one JSON object:

```json
{"text": "...", "source": "FEVER", "metadata": {"topic": "geography"}}
```

`text` is required. `source` and `metadata` are optional and default to the
filename's implied source.

## What's here now

A small, hand-written starter corpus (`fever.jsonl`, `halueval.jsonl`) so the
pipeline has *real* evidence to retrieve instead of the old two-topic demo
stub. It is intentionally broad but not exhaustive.

## Swapping in the real datasets

Replace or append to these files with the full datasets, keeping the same
`{"text": ..., "source": ..., "metadata": ...}` shape per line:

- **FEVER** (https://fever.ai/): use the `claim` field (or `claim` +
  `evidence` joined) as `text`.
- **HaluEval** (https://github.com/RUCAIBox/HaluEval): use the
  `right_answer` / `knowledge` field as `text` (not the hallucinated
  answer — this corpus is evidence, not examples of hallucinations).

No code changes are required — `KnowledgeBase.load()` reads every `*.jsonl`
file in this directory, embeds new/changed content, and rebuilds
`data/index/static.faiss` automatically (delete that file to force a full
rebuild after large data changes).

## Live Wikipedia retrieval

`services/knowledge_base.py` also queries the `wikipedia` PyPI package live,
per request, so the system isn't limited to whatever topics are pre-loaded
here. Wikipedia results are merged with the static corpus and re-ranked by
embedding similarity before the top-k evidence is handed to Ollama for
verification. If Wikipedia is unreachable (offline dev, rate limiting), the
system falls back to the static corpus only — it never fabricates a source.

## Future sources (PDF/document/web retrieval, MMHal-Bench, multimodal)

Add a new loader function alongside `_load_jsonl` /
`KnowledgeBase._live_wikipedia` in `services/knowledge_base.py` and merge its
results into `KnowledgeBase.retrieve()`. Nothing else in the pipeline needs
to change, since Stage 6 only depends on `KnowledgeBase.retrieve(query, k)`
returning `(RetrievalDocument, score)` pairs.
