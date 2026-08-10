"""Base class for pipeline stages.

A Stage is a pure(ish) unit of work: given a PipelineContext (and a bag
of services it needs), produce an updated PipelineContext. It does NOT
know about job IDs, DB timing, or status - that's execution/manager.py's
job. This separation is what makes each stage independently testable.
"""

from abc import ABC, abstractmethod

from models.enums import StageName
from pipeline.context import PipelineContext


class Stage(ABC):
    name: StageName
    # If False, a failure in this stage aborts the whole job. If True,
    # the manager logs it, marks the stage FAILED, and continues with
    # whatever the context already has (used for retrieval sources,
    # where "no evidence from source X" shouldn't kill the pipeline).
    critical: bool = True

    @abstractmethod
    async def run(self, context: PipelineContext) -> PipelineContext:
        ...
