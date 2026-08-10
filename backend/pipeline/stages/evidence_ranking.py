import logging

from config.settings import settings
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from retrieval.ranker import rank_evidence

logger = logging.getLogger(__name__)


class EvidenceRankingStage(Stage):
    name = StageName.EVIDENCE_RANKING
    critical = False

    async def run(self, context: PipelineContext) -> PipelineContext:
        for claim in context.claims:
            claim.evidence = rank_evidence(claim.text, claim.evidence, top_k=settings.retrieval_top_k)

        context.record(
            self.name.value,
            {"claims_with_evidence": sum(1 for c in context.claims if c.evidence)},
        )
        return context
