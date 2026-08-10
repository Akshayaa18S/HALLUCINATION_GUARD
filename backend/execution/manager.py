"""
Phase 4 - Pipeline Engine (+ Phase 12 optimizations: parallel retrieval,
caching is handled inside retrieval/cache.py, retry inside utils/retry.py).

Runs stages in order, wrapping each with start/finish timing recorded via
StageRepository. Non-critical stages (see Stage.critical) that raise are
logged and skipped rather than aborting the whole job - e.g. Wikipedia
being briefly unreachable shouldn't fail the entire analysis.

Wikipedia and FEVER retrieval run concurrently (both are pure network I/O
against independent sources). Their start/finish DB writes are kept
sequential around the concurrent section, since a single SQLAlchemy
AsyncSession isn't safe to use from two coroutines at once - only the
actual HTTP calls inside stage.run() are parallelized.
"""

import asyncio
import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from database.models import PipelineStage as PipelineStageRecord
from models.enums import JobStatus, StageStatus
from pipeline.context import PipelineContext
from pipeline.stages.base import Stage
from pipeline.stages.claim_extraction import ClaimExtractionStage
from pipeline.stages.coreference_resolution import CoreferenceResolutionStage
from pipeline.stages.entity_extraction import EntityExtractionStage
from pipeline.stages.evidence_ranking import EvidenceRankingStage
from pipeline.stages.explainability import ExplainabilityStage
from pipeline.stages.generation import GenerationStage
from pipeline.stages.hallucination_detection import HallucinationDetectionStage
from pipeline.stages.query_consistency import QueryConsistencyStage
from pipeline.stages.query_grounding import QueryGroundingStage
from pipeline.stages.retrieval_fever import FeverRetrievalStage
from pipeline.stages.retrieval_wikipedia import WikipediaRetrievalStage
from pipeline.stages.verification import VerificationStage
from services.job_service import JobService
from services.result_service import ResultRepository
from services.stage_service import StageRepository

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.job_service = JobService(db)
        self.stage_repo = StageRepository(db)
        self.result_repo = ResultRepository(db)

        # Sequential stages, in order. Retrieval stages are handled specially
        # (run concurrently) inside run_job(), so they're not in this list.
        self.query_grounding_stage = QueryGroundingStage()
        self.generation_stage = GenerationStage()
        self.claim_extraction_stage = ClaimExtractionStage()
        self.coreference_resolution_stage = CoreferenceResolutionStage()
        self.entity_extraction_stage = EntityExtractionStage()
        self.wikipedia_stage = WikipediaRetrievalStage()
        self.fever_stage = FeverRetrievalStage()
        self.evidence_ranking_stage = EvidenceRankingStage()
        self.verification_stage = VerificationStage()
        self.query_consistency_stage = QueryConsistencyStage()
        self.hallucination_stage = HallucinationDetectionStage()
        self.explainability_stage = ExplainabilityStage()

    async def _run_stage(self, context: PipelineContext, stage: Stage) -> PipelineContext:
        record = await self.stage_repo.start(context.job_id, stage.name.value)
        try:
            context = await stage.run(context)
        except Exception as exc:
            logger.error("Stage %s failed for job %s: %s", stage.name.value, context.job_id, exc)
            await self.stage_repo.finish(record, StageStatus.FAILED, error_message=str(exc))
            if stage.critical:
                raise
            return context
        else:
            metadata = context.stage_metadata.get(stage.name.value, {})
            await self.stage_repo.finish(record, StageStatus.COMPLETED, metadata=metadata)
            return context

    async def _run_retrieval_parallel(self, context: PipelineContext) -> PipelineContext:
        """Phase 12: Wikipedia + FEVER retrieval run concurrently."""
        wiki_record = await self.stage_repo.start(context.job_id, self.wikipedia_stage.name.value)
        fever_record = await self.stage_repo.start(context.job_id, self.fever_stage.name.value)

        results = await asyncio.gather(
            self.wikipedia_stage.run(context),
            self.fever_stage.run(context),
            return_exceptions=True,
        )

        for record, stage, outcome in (
            (wiki_record, self.wikipedia_stage, results[0]),
            (fever_record, self.fever_stage, results[1]),
        ):
            if isinstance(outcome, Exception):
                logger.error("Stage %s failed for job %s: %s", stage.name.value, context.job_id, outcome)
                await self.stage_repo.finish(record, StageStatus.FAILED, error_message=str(outcome))
                if stage.critical:
                    raise outcome
            else:
                metadata = context.stage_metadata.get(stage.name.value, {})
                await self.stage_repo.finish(record, StageStatus.COMPLETED, metadata=metadata)

        return context

    async def run_job(self, job_id: str, query: str) -> PipelineContext:
        start_time = time.monotonic()
        await self.job_service.update_status(job_id, JobStatus.RUNNING)
        context = PipelineContext(job_id=job_id, query=query)

        try:
            context = await self._run_stage(context, self.query_grounding_stage)
            context = await self._run_stage(context, self.generation_stage)
            context = await self._run_stage(context, self.claim_extraction_stage)
            context = await self._run_stage(context, self.coreference_resolution_stage)
            context = await self._run_stage(context, self.entity_extraction_stage)
            context = await self._run_retrieval_parallel(context)
            context = await self._run_stage(context, self.evidence_ranking_stage)
            context = await self._run_stage(context, self.verification_stage)
            context = await self._run_stage(context, self.query_consistency_stage)
            context = await self._run_stage(context, self.hallucination_stage)
            context = await self._run_stage(context, self.explainability_stage)
        except Exception as exc:
            await self.job_service.update_status(job_id, JobStatus.FAILED, error_message=str(exc))
            raise

        processing_time_ms = (time.monotonic() - start_time) * 1000

        def _dedup_evidence(ev_list):
            """Merge near-identical evidence snippets across sources, keeping best score.
            Uses both exact prefix matching and word-overlap (Jaccard ≥ 0.85) to catch
            duplicates from 'Career of X' vs the main biography page.
            """
            result = []
            for e in sorted(ev_list, key=lambda x: x.get("score", 0), reverse=True):
                txt_words = set(e.get("text", "").strip().lower().split())
                is_dup = False
                for kept in result:
                    kept_words = set(kept.get("text", "").strip().lower().split())
                    union = txt_words | kept_words
                    if union and len(txt_words & kept_words) / len(union) >= 0.85:
                        is_dup = True
                        break
                if not is_dup:
                    result.append(e)
            return result


        # Sort claims deterministically by text to guarantee payload ordering stability
        context.claims = sorted(context.claims, key=lambda c: (c.text.strip().lower(), c.verdict or ""))

        claims_payload = [
            {
                "text": c.text,
                "subject": c.subject,
                "relation": c.relation,
                "object": c.object,
                "sources": c.sources,
                "support_count": c.support_count,
                "contradiction_count": c.contradiction_count,
                "agreement": c.agreement,
                "evidence_quality": c.evidence_quality,
                "entities": [{"text": e.text, "label": e.label} for e in c.entities],
                "verdict": c.verdict,
                "confidence": c.confidence,
                "evidence": _dedup_evidence(c.evidence)[:3],
            }
            for c in context.claims
        ]





        await self.result_repo.create(
            job_id=job_id,
            generated_response=context.generated_response,
            verified_answer=context.verified_answer,
            explanation=context.explanation,
            overall_confidence=context.overall_confidence or 0.0,
            hallucination_score=context.hallucination_score or 0.0,
            processing_time_ms=processing_time_ms,
            claims=claims_payload,
        )
        await self.job_service.update_status(job_id, JobStatus.COMPLETED)

        logger.info("Job %s completed in %.1fms", job_id, processing_time_ms)
        return context