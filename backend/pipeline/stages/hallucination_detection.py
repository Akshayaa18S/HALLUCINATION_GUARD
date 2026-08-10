import logging

from hallucination.detector import ClaimVerdictInput, compute_hallucination_score
from models.enums import ClaimVerdict, StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


class HallucinationDetectionStage(Stage):
    name = StageName.HALLUCINATION_DETECTION
    critical = True

    async def run(self, context: PipelineContext) -> PipelineContext:
        inputs = [
            ClaimVerdictInput(
                verdict=ClaimVerdict(c.verdict),
                confidence=c.confidence,
                fabricated_alternative=c.fabricated_alternative,
            )
            for c in context.claims
            if c.verdict is not None
        ]
        score, confidence = compute_hallucination_score(inputs)
        context.hallucination_score = score
        context.overall_confidence = confidence

        context.record(self.name.value, {"hallucination_score": score, "confidence": confidence})
        return context
