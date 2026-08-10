"""
Phase 0 - Query Grounding (pre-generation retrieval).

Runs BEFORE generation. Extracts entities from the *user's query* - not
the LLM's response, that's entity_extraction.py's job - and retrieves
Wikipedia evidence for them, so the LLM can be handed real reference
material about the entities it's about to be asked about instead of
generating blind and only getting checked after the fact.

This is the fix for the most damaging failure mode this system can hit:
the model confidently denying a real, well-known entity exists at all
(and then inventing a similarly-named stand-in). Grounding evidence
found here is threaded through to two places:

  - generation.py includes it in the LLM's system prompt, so the model
    has a chance to get it right the first time.
  - query_consistency.py (post-verification) checks the response against
    it, so if the model denies an entity anyway despite the evidence
    sitting right there, that's caught as a contradiction rather than
    silently passed through because the "insufficient evidence" claims it
    invented instead happened to check out as unverifiable.

Kept as its own stage (rather than folded into entity_extraction) since
its input is the query, not a claim - a different object with a
different lifecycle, run at a different point in the pipeline.
"""

import logging

from config.settings import settings
from knowledge_base.ner import EntityRecognizer
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from retrieval.ranker import priority_order
from retrieval.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)

# Same reasoning as retrieval_wikipedia.py: dates/numbers aren't Wikipedia
# page topics and make poor search queries.
_NON_SEARCHABLE_ENTITY_LABELS = {"DATE", "NUMBER"}

# How many query entities to bother grounding. The query is usually short
# and about one or two things; anything past this is very likely noise.
_MAX_QUERY_ENTITIES = 3

# How much of a single entity's top evidence snippet to include verbatim
# in the generation prompt - long enough to be useful context, short
# enough not to dominate the prompt.
_SNIPPET_CHARS = 600


def _build_knowledge_context(evidence_by_entity: dict[str, list[dict]]) -> str:
    blocks = []
    for name, evidence in evidence_by_entity.items():
        if not evidence:
            continue
        snippet = evidence[0].get("text", "")[:_SNIPPET_CHARS]
        if snippet:
            blocks.append(f"{name}: {snippet}")
    return "\n\n".join(blocks)


class QueryGroundingStage(Stage):
    name = StageName.QUERY_GROUNDING
    critical = False  # no grounding evidence just means generation proceeds unaided, as before

    def __init__(
        self,
        recognizer: EntityRecognizer | None = None,
        retriever: WikipediaRetriever | None = None,
    ):
        self.recognizer = recognizer or EntityRecognizer()
        self.retriever = retriever or WikipediaRetriever()

    async def run(self, context: PipelineContext) -> PipelineContext:
        entities = self.recognizer.extract(context.query)
        searchable = [e for e in entities if e.label not in _NON_SEARCHABLE_ENTITY_LABELS]
        ordered = priority_order(searchable)[:_MAX_QUERY_ENTITIES]

        evidence_by_entity: dict[str, list[dict]] = {}
        hits = 0
        for entity in ordered:
            results = await self.retriever.retrieve(entity.text, top_k=settings.retrieval_top_k)
            evidence_by_entity[entity.text] = results
            hits += len(results)

        context.query_entities = entities
        context.query_evidence = evidence_by_entity
        context.knowledge_context = _build_knowledge_context(evidence_by_entity)

        context.record(
            self.name.value,
            {"query_entities": len(entities), "grounded_entities": len(evidence_by_entity), "evidence_found": hits},
        )
        return context
