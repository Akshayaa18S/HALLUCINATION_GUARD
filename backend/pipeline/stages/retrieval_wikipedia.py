import logging

from config.settings import settings
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from retrieval.ranker import priority_order
from retrieval.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)

# DATE and NUMBER entities (e.g. "thousands of years", "2020", "12%") aren't
# Wikipedia page topics - they're generic quantities that happen to appear
# inside a claim. Using one as the *sole* search query produces junk matches
# (e.g. "thousands of years" resolving to a film called "Three Thousand
# Years of Longing" instead of anything about domestication). They're still
# kept on claim.entities for display/verification purposes; they're just
# excluded from the candidate pool used to build Wikipedia queries.
_NON_SEARCHABLE_ENTITY_LABELS = {"DATE", "NUMBER"}


class WikipediaRetrievalStage(Stage):
    name = StageName.WIKIPEDIA_RETRIEVAL
    critical = False  # no wiki evidence just means a weaker verification, not a dead job

    def __init__(self, retriever: WikipediaRetriever | None = None, max_entities_per_claim: int = 3):
        self.retriever = retriever or WikipediaRetriever()
        self.max_entities_per_claim = max_entities_per_claim

    async def run(self, context: PipelineContext) -> PipelineContext:
        hits = 0
        for claim in context.claims:
            searchable_entities = [
                e for e in claim.entities if getattr(e, "label", None) not in _NON_SEARCHABLE_ENTITY_LABELS
            ]
            ordered = priority_order(searchable_entities)[: self.max_entities_per_claim]

            # Build disambiguated and contextual query terms
            query_terms = []
            seen_terms = set()

            subject_entity = next(
                (e.text for e in searchable_entities if getattr(e, "label", None) == "PERSON"), None
            )

            for e in ordered:
                if e.text not in seen_terms:
                    query_terms.append(e.text)
                    seen_terms.add(e.text)

                if subject_entity and e.text != subject_entity and len(e.text.split()) <= 2:
                    combo = f"{subject_entity} {e.text}"
                    if combo not in seen_terms:
                        query_terms.append(combo)
                        seen_terms.add(combo)

            clean_claim = claim.text.rstrip(".!?")
            if clean_claim and clean_claim not in seen_terms:
                query_terms.append(clean_claim)
                seen_terms.add(clean_claim)

            for term in query_terms:
                results = await self.retriever.retrieve(term, top_k=settings.retrieval_top_k)
                claim.evidence.extend(results)
                hits += len(results)

        context.record(self.name.value, {"evidence_found": hits})
        return context