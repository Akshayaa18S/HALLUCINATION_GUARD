import logging

from models.enums import StageName
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Deliberately instructs the model NOT to paper over a gap in its
# knowledge with a plausible-sounding invention - the exact failure mode
# (deny the real entity, then fabricate a similarly-named one) that
# query_grounding.py + query_consistency.py exist to catch when it slips
# through anyway.
_BASE_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Only state facts you are "
    "confident about, and never invent a plausible-sounding alternative "
    "(a different person, place, or thing with a similar name) to stand "
    "in for something you don't actually know. If you're not confident "
    "about a specific detail, say so plainly instead of guessing."
)

_GROUNDED_SYSTEM_PROMPT_SUFFIX = (
    "\n\nThe reference information below was retrieved specifically for "
    "this question. Treat it as ground truth about the entity's existence "
    "and identity - if it describes an entity mentioned in the question, "
    "that entity is real, so do not claim no such entity exists.\n\n"
    "REFERENCE INFORMATION:\n{knowledge_context}"
)


class GenerationStage(Stage):
    name = StageName.GENERATION
    critical = True  # no response, nothing else can run

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def run(self, context: PipelineContext) -> PipelineContext:
        system = _BASE_SYSTEM_PROMPT
        if context.knowledge_context:
            system += _GROUNDED_SYSTEM_PROMPT_SUFFIX.format(knowledge_context=context.knowledge_context)

        response = await self.llm_service.generate(context.query, system=system)
        context.generated_response = response
        context.record(
            self.name.value,
            {"response_length": len(response), "used_knowledge_context": bool(context.knowledge_context)},
        )
        return context
