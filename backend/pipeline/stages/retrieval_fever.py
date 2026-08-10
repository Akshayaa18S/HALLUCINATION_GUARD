import logging

from config.settings import settings
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from retrieval.fever_retriever import FeverRetriever

logger = logging.getLogger(__name__)


class FeverRetrievalStage(Stage):
    name = StageName.FEVER_RETRIEVAL
    critical = False

    def __init__(self, retriever: FeverRetriever | None = None):
        self.retriever = retriever or FeverRetriever()

    async def run(self, context: PipelineContext) -> PipelineContext:
        hits = 0
        for claim in context.claims:
            results = await self.retriever.retrieve(claim.text, top_k=settings.retrieval_top_k)
            claim.evidence.extend(results)
            hits += len(results)

        context.record(self.name.value, {"evidence_found": hits})
        return context
