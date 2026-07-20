import { AnalysisResult, PIPELINE_STAGES, StageStatus } from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || API_BASE_URL.replace(/^http/, "ws");

export interface StageProgressEvent {
  message_type: "stage_progress" | "result" | "error";
  data: any;
  timestamp: string;
}

export interface RunHandlers {
  onStageUpdate: (stageNumber: number, status: StageStatus, durationMs?: number) => void;
  onResult: (result: AnalysisResult) => void;
  onError: (message: string) => void;
}

/**
 * Submit an analysis job to the real backend and stream progress over WebSocket.
 * Falls back to a local simulation (so the UI is still demoable without a
 * running backend) if the REST call fails outright, e.g. no server reachable.
 */
export async function runAnalysis(
  query: string,
  mode: "text" | "image" | "both",
  handlers: RunHandlers
): Promise<() => void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_text: query }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const job = await response.json();
    const jobId: string = job.job_id;

    const ws = new WebSocket(`${WS_BASE_URL}/ws/progress/${jobId}`);

    ws.onmessage = (event) => {
      try {
        const payload: StageProgressEvent = JSON.parse(event.data);
        handleEvent(payload, handlers);
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      handlers.onError("Lost connection to the analysis server.");
    };

    return () => ws.close();
  } catch (err) {
    // No backend reachable — run a local simulation so the dashboard still works.
    return simulateAnalysis(query, handlers);
  }
}

function handleEvent(payload: StageProgressEvent, handlers: RunHandlers) {
  if (payload.message_type === "stage_progress") {
    const { stage, status, duration_ms } = payload.data;
    handlers.onStageUpdate(stage, status === "completed" ? "completed" : "running", duration_ms);
  } else if (payload.message_type === "result") {
    const d = payload.data;
    handlers.onResult({
      job_id: d.job_id,
      generated_response: d.generated_response,
      verified_answer: d.verified_answer,
      hallucination: d.hallucination,
      confidence: d.confidence,
      retrieved_evidence: d.retrieved_evidence || {},
      explanation: d.explanation,
      processing_time_ms: d.processing_time_ms,
    });
  } else if (payload.message_type === "error") {
    handlers.onError(payload.data?.error_message || "Analysis failed.");
  }
}

/** Deterministic offline demo so the UI is fully explorable without a backend running. */
function simulateAnalysis(query: string, handlers: RunHandlers): () => void {
  let cancelled = false;
  const timeouts: ReturnType<typeof setTimeout>[] = [];

  const isTelephone = /telephone/i.test(query);
  const generated = isTelephone
    ? "The telephone was invented by Thomas Edison in 1876."
    : `Response generated for: "${query}"`;
  const verified = isTelephone
    ? "Alexander Graham Bell invented the telephone."
    : "The claim could not be fully verified against known sources.";
  const hallucinated = isTelephone ? true : Math.random() > 0.5;

  let elapsed = 0;
  PIPELINE_STAGES.forEach((stage, i) => {
    const runDelay = elapsed + 150;
    const doneDelay = runDelay + 350 + i * 60;
    elapsed = doneDelay;

    timeouts.push(
      setTimeout(() => {
        if (!cancelled) handlers.onStageUpdate(stage.number, "running");
      }, runDelay)
    );
    timeouts.push(
      setTimeout(() => {
        if (!cancelled) handlers.onStageUpdate(stage.number, "completed", doneDelay - runDelay);
      }, doneDelay)
    );
  });

  timeouts.push(
    setTimeout(() => {
      if (cancelled) return;
      handlers.onResult({
        job_id: `job_${Math.random().toString(36).slice(2, 10)}`,
        generated_response: generated,
        verified_answer: verified,
        hallucination: hallucinated,
        confidence: hallucinated ? 0.92 : 0.11,
        retrieved_evidence: isTelephone
          ? {
              supporting_documents: ["Wikipedia - Alexander Graham Bell", "FEVER Dataset"],
              evidence: [
                "Alexander Graham Bell is credited with inventing the telephone. He patented the first practical telephone in 1876.",
              ],
              contradictions: [
                `The claim '${generated}' contradicts established historical records.`,
              ],
            }
          : { supporting_documents: ["Wikipedia", "FEVER Dataset"], evidence: [], contradictions: [] },
        explanation: isTelephone
          ? "The generated response incorrectly states that Thomas Edison invented the telephone. According to verified sources, Alexander Graham Bell is credited with inventing the telephone."
          : "The claim was evaluated against retrieved evidence.",
        processing_time_ms: elapsed,
        tokens_input: Math.max(8, Math.round(query.length / 4)),
        tokens_output: 28,
      });
    }, elapsed + 200)
  );

  return () => {
    cancelled = true;
    timeouts.forEach(clearTimeout);
  };
}
