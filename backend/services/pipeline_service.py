"""
Pipeline orchestration service for the 8-stage hallucination detection flow.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from config import settings
from models.job import Job, JobStatus
from models.result import Result
from models.stage import Stage, StageStatus
from services.job_manager import JobManager
from services import llm_service
from services.ollama_service import OllamaService, ollama_service
from services.knowledge_base import knowledge_base
from services.verification_service import verification_service
from services.hallucination_service import hallucination_service
from services.explainability_service import explainability_service
# MultiHaluDet base-paper branch: local in-process generation (needed
# for hidden-state/logit access - Ollama's HTTP API can't provide that)
# plus the hidden-state trajectory probing pipeline itself. See
# multihaludet/__init__.py for the full stage breakdown.
from multihaludet.generation_backend import GenerationBundle
from multihaludet.service import multihaludet_service
from schemas.event_schemas import (
    StageEvent,
    FinalResultEvent,
    WebSocketMessage,
    ErrorEvent,
)
from utils.logging_config import (
    log_job_complete,
    log_job_error,
    log_stage_complete,
    log_stage_error,
    log_stage_start,
    logger,
)

StageCallback = Optional[Callable[[Dict[str, Any]], Any]]


class PipelineService:
    """Executes the staged hallucination detection pipeline."""

    STAGES = [
        (1, "Input Received"),
        (2, "Generating Response"),
        (3, "Hidden State Extraction"),
        (4, "Feature Extraction"),
        (5, "Hallucination Detection"),
        (6, "Fact Verification"),
        (7, "Explainability"),
        (8, "Analysis Completed"),
    ]

    def __init__(self, db: Session, dev_mode: Optional[bool] = None):
        self.db = db
        self.dev_mode = settings.DEV_MODE if dev_mode is None else dev_mode
        self.max_stage_retries = max(1, getattr(settings, "JOB_MAX_RETRIES", 3))
        self.ollama_service = ollama_service
        self.llm_service = llm_service
        self.knowledge_base = knowledge_base
        self.verification_service = verification_service
        self.hallucination_service = hallucination_service
        self.explainability_service = explainability_service
        self.multihaludet_service = multihaludet_service

    async def execute(self, job_id: str, progress_callback: StageCallback = None) -> Dict[str, Any]:
        """Run the pipeline for a job and persist all stage outputs."""
        job = JobManager.get_job(self.db, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        started_at = time.perf_counter()
        JobManager.update_job_status(self.db, job_id, JobStatus.RUNNING.value)

        pipeline_state: Dict[str, Any] = {
            "job_id": job_id,
            "input_text": job.input_text,
            "input_image_path": job.input_image_path,
            "user_query": job.input_text or (f"Image analysis request for {job.input_image_path}" if job.input_image_path else "the provided input"),
            "generated_response": None,
            "hidden_states": {},
            "extracted_features": {},
            "hallucination_result": {},
            "retrieved_evidence": {},
            "explanation": "",
            "execution_pipeline": [],
        }

        try:
            await self._run_stage(
                job,
                1,
                "Input Received",
                10,
                progress_callback,
                pipeline_state,
                self._stage_1,
            )
            await self._run_stage(
                job,
                2,
                "Generating Response",
                20,
                progress_callback,
                pipeline_state,
                self._stage_2,
            )
            await self._run_stage(
                job,
                3,
                "Hidden State Extraction",
                35,
                progress_callback,
                pipeline_state,
                self._stage_3,
            )
            await self._run_stage(
                job,
                4,
                "Feature Extraction",
                50,
                progress_callback,
                pipeline_state,
                self._stage_4,
            )
            await self._run_stage(
                job,
                5,
                "Hallucination Detection",
                70,
                progress_callback,
                pipeline_state,
                self._stage_5,
            )
            await self._run_stage(
                job,
                6,
                "Fact Verification",
                85,
                progress_callback,
                pipeline_state,
                self._stage_6,
            )
            await self._run_stage(
                job,
                7,
                "Explainability",
                95,
                progress_callback,
                pipeline_state,
                self._stage_7,
            )
            await self._run_stage(
                job,
                8,
                "Analysis Completed",
                100,
                progress_callback,
                pipeline_state,
                self._stage_8,
            )

            total_processing_time_ms = (time.perf_counter() - started_at) * 1000
            result = self._persist_result(job_id, pipeline_state, total_processing_time_ms)
            JobManager.update_job_status(self.db, job_id, JobStatus.COMPLETED.value)
            log_job_complete(job_id, total_processing_time_ms)

            if progress_callback:
                await self._maybe_call_progress(progress_callback, self._build_final_message(result, total_processing_time_ms))

            return {
                "completed": True,
                "hallucination": result.is_hallucination == "yes",
                "confidence": result.confidence,
                "generated_response": result.generated_response,
                "verified_answer": result.verified_answer,
                "retrieved_evidence": result.retrieved_evidence,
                "explanation": result.explanation_text,
                "user_query": result.user_query,
                "execution_pipeline": result.execution_pipeline,
                "processing_time": f"{total_processing_time_ms / 1000:.2f} sec",
            }
        except Exception as exc:
            log_job_error(job_id, str(exc))
            JobManager.update_job_status(self.db, job_id, JobStatus.FAILED.value, str(exc))
            raise

    async def _run_stage(
        self,
        job: Job,
        stage_number: int,
        stage_name: str,
        progress: int,
        progress_callback: StageCallback,
        pipeline_state: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
        completed_stage: bool = False,
    ) -> None:
        stage = self._start_stage(job.job_id, stage_number, stage_name, progress, progress_callback)
        if progress_callback:
            await self._maybe_call_progress(
                progress_callback, self._build_stage_message(stage, "running", progress, {})
            )
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_stage_retries:
            attempt += 1
            try:
                # Apply configurable delay simulation in dev mode
                if self.dev_mode and settings.DELAY_SIMULATION_ENABLED:
                    await asyncio.sleep(random.uniform(settings.STAGE_DELAY_MIN_MS / 1000, settings.STAGE_DELAY_MAX_MS / 1000))

                # Call handler; support coroutine functions and sync functions
                if inspect.iscoroutinefunction(handler):
                    stage_output = await handler(pipeline_state)
                else:
                    # run sync handler in thread to avoid blocking
                    stage_output = await asyncio.to_thread(handler, pipeline_state)

                if not isinstance(stage_output, dict):
                    stage_output = {} if stage_output is None else {"result": stage_output}

                pipeline_state.update(stage_output)
                self._complete_stage(stage, progress, stage_output)
                if stage_number == 8:
                    # Build the persisted snapshot after Stage 8 itself is
                    # completed, preventing the client from seeing it as running.
                    pipeline_state["execution_pipeline"] = self._build_execution_pipeline(pipeline_state)
                    stage_output["execution_pipeline"] = pipeline_state["execution_pipeline"]
                    stage.metadata_json = {"analysis_completed": True, "execution_pipeline": pipeline_state["execution_pipeline"]}
                    self.db.commit()
                if progress_callback:
                    await self._maybe_call_progress(progress_callback, self._build_stage_message(stage, "completed", progress, stage_output))
                return
            except Exception as exc:
                last_error = exc
                logger.warning(f"Stage {stage_number} attempt {attempt} failed: {str(exc)}")
                # if we can retry, wait with exponential backoff
                if attempt < self.max_stage_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                # final failure
                self._fail_stage(stage, str(exc))
                if progress_callback:
                    await self._maybe_call_progress(progress_callback, self._build_stage_message(stage, "failed", progress, {}, str(exc)))
                raise

    def _start_stage(
        self,
        job_id: str,
        stage_number: int,
        stage_name: str,
        progress: int,
        progress_callback: StageCallback,
    ) -> Stage:
        stage = Stage(
            job_id=job_id,
            stage_number=stage_number,
            name=stage_name,
            status=StageStatus.RUNNING.value,
            progress_percentage=float(progress),
            start_time=datetime.utcnow(),
            metadata_json={},
        )
        self.db.add(stage)
        self.db.commit()
        self.db.refresh(stage)
        log_stage_start(job_id, stage_number, stage_name)
        return stage

    async def _maybe_call_progress(self, cb: StageCallback, payload: Dict[str, Any]) -> None:
        try:
            if not cb:
                return
            result = cb(payload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning(f"Progress callback raised an exception: {str(exc)}")

    def _complete_stage(self, stage: Stage, progress: int, metadata: Dict[str, Any]) -> None:
        stage.status = StageStatus.COMPLETED.value
        stage.progress_percentage = float(progress)
        stage.end_time = datetime.utcnow()
        stage.duration_ms = int((stage.end_time - stage.start_time).total_seconds() * 1000) if stage.start_time else None
        stage.metadata_json = metadata
        self.db.commit()
        self.db.refresh(stage)
        log_stage_complete(stage.job_id, stage.stage_number, stage.name, stage.duration_ms or 0)

    def _fail_stage(self, stage: Stage, error_message: str) -> None:
        stage.status = StageStatus.FAILED.value
        stage.error_message = error_message
        stage.end_time = datetime.utcnow()
        stage.duration_ms = int((stage.end_time - stage.start_time).total_seconds() * 1000) if stage.start_time else None
        self.db.commit()
        self.db.refresh(stage)
        log_stage_error(stage.job_id, stage.stage_number, stage.name, error_message)

    def _build_stage_message(
        self,
        stage: Stage,
        status: str,
        progress: int,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        stage_event = StageEvent(
            job_id=stage.job_id,
            stage=stage.stage_number,
            name=stage.name,
            status=status,
            progress_percentage=progress,
            start_time=stage.start_time,
            end_time=stage.end_time,
            duration_ms=stage.duration_ms,
            metadata=metadata or stage.metadata_json or {},
            error_message=error_message,
        )
        return WebSocketMessage(message_type="stage_progress", data=stage_event.dict()).dict()

    def _build_final_message(self, result: Result, total_processing_time_ms: float) -> Dict[str, Any]:
        final_event = FinalResultEvent(
            job_id=result.job_id,
            status="completed",
            hallucination=result.is_hallucination == "yes",
            confidence=result.confidence,
            generated_response=result.generated_response,
            verified_answer=result.verified_answer,
            retrieved_evidence=result.retrieved_evidence,
            explanation=result.explanation_text,
            processing_time_ms=total_processing_time_ms,
        )
        return WebSocketMessage(message_type="result", data=final_event.dict()).dict()

    def _build_error_message(self, job_id: str, stage_number: Optional[int], error_message: str) -> Dict[str, Any]:
        error_event = ErrorEvent(
            job_id=job_id,
            stage=stage_number,
            error_message=error_message,
        )
        return WebSocketMessage(message_type="error", data=error_event.dict()).dict()

    async def _stage_1(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        state["user_query"] = query
        return {
            "input_received": True,
            "input_type": self._input_type(state),
            "user_query": query,
        }

    async def _stage_2(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        has_image = bool(state.get("input_image_path"))
        prompt = query.strip() or "the provided input"
        if has_image:
            prompt = f"{prompt}\n\n(Note: an image was attached to this request, but only the text is visible here.)"

        # Blocking torch call - keep it off the event loop. This is the
        # SAME forward pass stage 3/4 read hidden states from below, so
        # the internal signal reflects the response actually returned,
        # not a second, separately-sampled generation.
        bundle: GenerationBundle = await asyncio.to_thread(self.multihaludet_service.generate, prompt)

        state["_generation_bundle"] = bundle
        state["generated_response"] = bundle.text
        return {
            "generated_response": bundle.text,
            "generation_backend": "multihaludet_local_hf",
            "prompt_token_count": bundle.prompt_token_count,
            "generated_token_count": int(bundle.step_logits.shape[0]),
        }

    def _get_multihaludet_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run (and cache) the full MultiHaluDet forward pass for this job's
        generation bundle. Stage 3 and stage 4 report different slices of
        the same result rather than recomputing it twice."""
        cached = state.get("_multihaludet_result")
        if cached is not None:
            return cached
        bundle: GenerationBundle | None = state.get("_generation_bundle")
        if bundle is None:
            # Defensive: should not happen since stage 2 always sets this,
            # but never crash the pipeline over a missing internal cache.
            result: Dict[str, Any] = {
                "internal_hallucination_probability": 0.5,
                "internal_confidence": 0.0,
                "is_trained": False,
                "note": "no_generation_bundle",
                "selected_layers": [],
                "num_total_layers": 0,
                "generated_tokens": 0,
                "ensemble_member_names": [],
                "ensemble_member_probabilities": {},
                "layer_importance_weights": [],
                "self_attention_pooling_weights": [],
                "multi_scale_gate_weights": [],
                "global_feature_names": [],
                "global_feature_values": [],
            }
        else:
            result = self.multihaludet_service.score(bundle)
        state["_multihaludet_result"] = result
        return result

    async def _stage_3(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Dynamic/multi-depth layer sampling + sequential representation:
        # real hidden-state statistics from the local model's forward
        # pass (multihaludet/layer_sampling.py), not placeholder numbers.
        result = await asyncio.to_thread(self._get_multihaludet_result, state)
        hidden_states = {
            "selected_layers": result["selected_layers"],
            "num_total_layers": result["num_total_layers"],
            "generated_token_count": result["generated_tokens"],
            "layer_norm_trajectory": {
                name: value
                for name, value in zip(result.get("global_feature_names", []), result.get("global_feature_values", []))
                if name.startswith("layer_norm") or name.startswith("anchor_")
            },
        }
        state["hidden_states"] = hidden_states
        return hidden_states

    async def _stage_4(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Multi-scale attention -> layer-weighted Transformer encoder ->
        # self-attention pooling -> global branch -> gated fusion, i.e.
        # the rest of the base paper's architecture, applied to the same
        # cached forward pass from stage 3.
        result = await asyncio.to_thread(self._get_multihaludet_result, state)
        extracted_features = {
            "multi_scale_attention_gate_weights": result["multi_scale_gate_weights"],
            "layer_weighted_transformer_importance": result["layer_importance_weights"],
            "self_attention_pooling_weights": result["self_attention_pooling_weights"],
            "global_branch_gate_mean": result.get("global_branch_gate_mean"),
            "global_feature_summary": dict(
                zip(result.get("global_feature_names", []), result.get("global_feature_values", []))
            ),
            "internal_hallucination_probability": result["internal_hallucination_probability"],
            "internal_confidence": result["internal_confidence"],
            "model_is_trained": result["is_trained"],
            "model_checkpoint_path": result.get("checkpoint_path"),
        }
        state["extracted_features"] = extracted_features
        return extracted_features

    async def _stage_5(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Signal 1 (external): does retrieved evidence support/contradict
        # the generated claims? (RAG/evidence-verification branch)
        fact_check = self._get_fact_check(state)
        external_result = self.hallucination_service.score(
            self._normalize_query(state),
            state.get("generated_response", ""),
            fact_check,
            fact_check.get("evidence") or [],
        )

        # Signal 2 (internal): what does the LLM's own hidden-state
        # trajectory suggest? (MultiHaluDet branch, stages 3-4 above)
        internal_result = self._get_multihaludet_result(state)

        hallucination_result = self._fuse_internal_and_external(internal_result, external_result)
        state["hallucination_result"] = hallucination_result
        return hallucination_result

    def _fuse_internal_and_external(
        self, internal_result: Dict[str, Any], external_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dual-signal fusion: combine MultiHaluDet's internal probability
        with the RAG/evidence branch's external probability. A simple,
        transparent weighted average by design - `fusion_internal_weight`
        in config/settings.py is the one knob, so the fusion behavior is
        auditable rather than a second opaque model. Confidence is
        similarly a weighted blend, discounted whenever the two signals
        disagree strongly (that disagreement is itself informative and
        shouldn't be averaged away silently)."""
        w = max(0.0, min(1.0, settings.fusion_internal_weight))
        internal_p = float(internal_result.get("internal_hallucination_probability", 0.5))
        external_p = float(external_result.get("probability", 0.5))
        fused_probability = w * internal_p + (1 - w) * external_p

        internal_conf = float(internal_result.get("internal_confidence", 0.0))
        external_conf = float(external_result.get("confidence", 0.0))
        disagreement = abs(internal_p - external_p)
        fused_confidence = max(0.0, (w * internal_conf + (1 - w) * external_conf) * (1 - disagreement))

        if fused_probability >= 0.66:
            decision, label = "yes", "high"
        elif fused_probability <= 0.33:
            decision, label = "no", "low"
        else:
            decision, label = external_result.get("decision", "uncertain"), "uncertain"

        result = dict(external_result)
        result.update({
            "prediction": decision == "yes",
            "decision": decision,
            "label": label,
            "hallucination_probability": round(fused_probability, 4),
            "probability": round(fused_probability, 4),
            "confidence": round(fused_confidence, 4),
            "internal_score": round(internal_p, 4),
            "internal_confidence": round(internal_conf, 4),
            "external_score": round(external_p, 4),
            "external_confidence": round(external_conf, 4),
            "fusion_internal_weight": w,
            "model_votes": internal_result.get("ensemble_member_probabilities", {}),
            "model_is_trained": internal_result.get("is_trained", False),
            "stacking": "dual_signal_fusion(multihaludet_internal, rag_external)",
        })
        return result

    async def _stage_6(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        generated_response = state.get("generated_response", "") or ""
        fact_check = self._get_fact_check(state)
        verified_answer = fact_check.get("verified_answer") or generated_response or query

        # `evidence` / `supporting_documents` / `contradictions` are lists of
        # structured {source, content, score} objects — never placeholder
        # strings. An empty list means no relevant evidence was found, and
        # that is reported honestly rather than backfilled with fake text.
        evidence = list(fact_check.get("evidence") or [])
        contradictions = list(fact_check.get("contradictions") or [])
        supporting_documents = list(fact_check.get("supporting_documents") or [])
        sources = sorted({doc.get("source") for doc in supporting_documents if doc.get("source")})

        retrieved_evidence = {
            "sources": sources,
            "supporting_documents": supporting_documents,
            "evidence": evidence,
            "contradictions": contradictions,
        }
        state["retrieved_evidence"] = retrieved_evidence
        state["verified_answer"] = verified_answer
        return retrieved_evidence

    async def _stage_7(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        fact_check = self._get_fact_check(state)
        verified_answer = state.get("verified_answer") or fact_check.get("verified_answer") or ""
        evidence = state.get("retrieved_evidence", {}).get("evidence", [])
        explanation_text = self.explainability_service.explain(
            query,
            state.get("generated_response", ""),
            fact_check,
            state.get("hallucination_result", {}),
            evidence,
        )

        internal_result = self._get_multihaludet_result(state)
        hallucination_result = state.get("hallucination_result", {})
        internal_note = (
            "MultiHaluDet branch is architecture-only (no trained checkpoint "
            "loaded) - internal_score reflects randomly-initialized weights, "
            "not a validated hallucination judgment."
            if not internal_result.get("is_trained")
            else "MultiHaluDet branch is running a trained checkpoint."
        )
        explanation = {
            "shap_values": [round(0.1 + index * 0.12, 2) for index in range(3)],
            "important_tokens": self._tokenize(query)[:5],
            "attention_heatmap": "generated/in-memory-heatmap.png",
            "explanation_text": explanation_text,
            "internal_signal_summary": {
                "internal_score": hallucination_result.get("internal_score"),
                "external_score": hallucination_result.get("external_score"),
                "fusion_internal_weight": hallucination_result.get("fusion_internal_weight"),
                "top_layer_importance": sorted(
                    internal_result.get("layer_importance_weights", []), reverse=True
                )[:3],
                "note": internal_note,
            },
        }
        state["explanation"] = explanation
        return explanation

    async def _stage_8(self, state: Dict[str, Any]) -> Dict[str, Any]:
        execution_pipeline = self._build_execution_pipeline(state)
        state["execution_pipeline"] = execution_pipeline
        return {"analysis_completed": True, "execution_pipeline": execution_pipeline}

    def _persist_result(self, job_id: str, state: Dict[str, Any], total_processing_time_ms: float) -> Result:
        hallucination_result = state.get("hallucination_result", {}) or {}
        retrieved_evidence = state.get("retrieved_evidence", {}) or {}
        verified_answer = (
            state.get("verified_answer")
            or (state.get("_fact_check") or {}).get("verified_answer")
            or state.get("generated_response")
        )
        probability = hallucination_result.get("probability", 0.0)
        confidence = hallucination_result.get("confidence", 0.0)
        is_hallucination = str(hallucination_result.get("decision", "uncertain"))

        result = Result(
            job_id=job_id,
            user_query=state.get("user_query") or state.get("input_text") or state.get("input_image_path"),
            hallucination_score=float(probability * 100),
            confidence=float(confidence) if confidence else 0.0,
            is_hallucination=is_hallucination,
            generated_response=state.get("generated_response"),
            hidden_states=state.get("hidden_states"),
            extracted_features=state.get("extracted_features"),
            retrieved_evidence=retrieved_evidence,
            supporting_documents=retrieved_evidence.get("supporting_documents"),
            contradictions=retrieved_evidence.get("contradictions"),
            verified_answer=verified_answer,
            shap_explanation=state.get("explanation", {}).get("shap_values"),
            important_tokens=state.get("explanation", {}).get("important_tokens"),
            attention_heatmap=state.get("explanation", {}).get("attention_heatmap"),
            explanation_text=state.get("explanation", {}).get("explanation_text"),
            execution_pipeline=state.get("execution_pipeline") or self._build_execution_pipeline(state),
            total_processing_time_ms=total_processing_time_ms,
        )
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def _get_fact_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run (and cache) the evidence-driven fact-check/hallucination verdict for this job."""
        cached = state.get("_fact_check")
        if cached is not None:
            return cached
        query = self._normalize_query(state)
        generated_response = state.get("generated_response", "") or ""

        fact_check = self.verification_service.verify(query, generated_response)
        state["_fact_check"] = fact_check
        return fact_check

    def _normalize_query(self, state: Dict[str, Any]) -> str:
        if state.get("input_text"):
            return str(state.get("input_text")).strip()
        if state.get("input_image_path"):
            return f"Image analysis request for {state.get('input_image_path')}"
        return "the provided input"

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[\w']+\b", text.lower())

    def _build_execution_pipeline(self, state: Dict[str, Any]) -> list[dict[str, Any]]:
        stages = (
            self.db.query(Stage)
            .filter(Stage.job_id == state.get("job_id"))
            .order_by(Stage.stage_number.asc())
            .all()
        )
        return [
            {
                "stage": stage.name,
                "status": stage.status,
                "time_ms": stage.duration_ms or 0,
            }
            for stage in stages
        ]

    def _input_type(self, state: Dict[str, Any]) -> str:
        if state.get("input_text") and state.get("input_image_path"):
            return "text_image"
        if state.get("input_image_path"):
            return "image"
        return "text"
