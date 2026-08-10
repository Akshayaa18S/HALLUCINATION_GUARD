import logging

from knowledge_base.ner import EntityRecognizer
from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage

logger = logging.getLogger(__name__)


class EntityExtractionStage(Stage):
    name = StageName.ENTITY_EXTRACTION
    critical = False  # claims with no entities still get verified via generic context retrieval

    def __init__(self, recognizer: EntityRecognizer | None = None):
        self.recognizer = recognizer or EntityRecognizer()

    async def run(self, context: PipelineContext) -> PipelineContext:
        total_entities = 0
        for claim in context.claims:
            claim.entities = self.recognizer.extract(claim.text)
            total_entities += len(claim.entities)
        context.record(self.name.value, {"total_entities": total_entities})
        return context
