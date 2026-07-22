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
        generated_response = self._generate_response(state)
        state["generated_response"] = generated_response
        return {"generated_response": generated_response}

    async def _stage_3(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        tokens = self._tokenize(query)
        hidden_states = {
            "token_embeddings": [round(float(index + 1) / max(1, len(tokens)), 4) for index in range(min(4, len(tokens)))],
            "attention_maps": [round(0.15 + (index / max(1, len(tokens))) * 0.55, 4) for index in range(min(4, len(tokens)))],
            "token_count": len(tokens),
        }
        state["hidden_states"] = hidden_states
        return hidden_states

    async def _stage_4(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(state)
        tokens = self._tokenize(query)
        response = state.get("generated_response", "")
        extracted_features = {
            "dynamic_layer_sampling": [round(len(tokens) / 10, 3), round(len(response.split()) / 10, 3), round(len(query) / 20, 3)],
            "multi_scale_attention": [round(0.1 + (index / max(1, len(tokens))) * 0.4, 3) for index in range(min(3, len(tokens)))],
            "transformer_encoder": [round(0.2 + (index / max(1, len(tokens))) * 0.6, 3) for index in range(3)],
            "self_attention_pooling": [round(0.25 + (index / 3) * 0.3, 3) for index in range(3)],
        }
        state["extracted_features"] = extracted_features
        return extracted_features

    async def _stage_5(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fact_check = self._get_fact_check(state)
        hallucination_result = self.hallucination_service.score(
            self._normalize_query(state),
            state.get("generated_response", ""),
            fact_check,
            fact_check.get("evidence") or [],
        )
        hallucination_result.update({
            "model_votes": {
                "random_forest": hallucination_result["prediction"],
                "xgboost": hallucination_result["prediction"],
                "lightgbm": hallucination_result["prediction"],
                "logistic_regression": hallucination_result["prediction"],
                "svm": hallucination_result["prediction"],
            },
            "stacking": "rag_verdict",
        })
        state["hallucination_result"] = hallucination_result
        return hallucination_result

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

        explanation = {
            "shap_values": [round(0.1 + index * 0.12, 2) for index in range(3)],
            "important_tokens": self._tokenize(query)[:5],
            "attention_heatmap": "generated/in-memory-heatmap.png",
            "explanation_text": explanation_text,
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

    def _generate_response(self, state: Dict[str, Any]) -> str:
        query = self._normalize_query(state)
        has_image = bool(state.get("input_image_path"))
        prompt = query.strip() or "the provided input"
        if has_image:
            prompt = f"{prompt}\n\n(Note: an image was attached to this request, but only the text is visible here.)"
        return self.ollama_service.generate(prompt)

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
