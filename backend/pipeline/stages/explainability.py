import logging

from hallucination.explainability import ClaimExplainInput, build_explanation
from models.enums import ClaimVerdict, StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


class ExplainabilityStage(Stage):
    name = StageName.EXPLAINABILITY
    critical = False  # worst case: no polished explanation text, job still succeeds

    async def run(self, context: PipelineContext) -> PipelineContext:
        inputs = [
            ClaimExplainInput(
                text=c.text,
                verdict=ClaimVerdict(c.verdict) if c.verdict else ClaimVerdict.INSUFFICIENT,
                confidence=c.confidence,
                evidence=c.evidence,
                fabricated_alternative=c.fabricated_alternative,
            )
            for c in context.claims
        ]
        result = build_explanation(inputs)
        context.verified_answer = result.verified_answer
        context.explanation = result.explanation
        context.contradictions = result.contradictions

        context.record(self.name.value, {"contradiction_count": len(result.contradictions)})
        return context
