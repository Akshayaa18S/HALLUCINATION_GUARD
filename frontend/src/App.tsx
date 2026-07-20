import { useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import InputPanel from "./components/InputPanel";
import ModelInfoPanel from "./components/ModelInfoPanel";
import ExecutionPipeline from "./components/ExecutionPipeline";
import ResultTabs from "./components/ResultTabs";
import OverallResultPanel from "./components/OverallResultPanel";
import RetrievedEvidencePanel from "./components/RetrievedEvidencePanel";
import RequestDetailsPanel from "./components/RequestDetailsPanel";
import { runAnalysis } from "./api";
import { AnalysisResult, InputMode, PIPELINE_STAGES, StageState, StageStatus } from "./types";

const initialStages: StageState[] = PIPELINE_STAGES.map((s) => ({ ...s, status: "pending" as StageStatus }));

export default function App() {
  const [mode, setMode] = useState<InputMode>("text");
  const [query, setQuery] = useState("Who invented the telephone?");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stages, setStages] = useState<StageState[]>(initialStages);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [totalTimeMs, setTotalTimeMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const startRef = useRef<number>(0);

  const handleSubmit = async () => {
    if (!query.trim() || isAnalyzing) return;

    cleanupRef.current?.();
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    setTotalTimeMs(null);
    setStages(initialStages);
    startRef.current = performance.now();

    cleanupRef.current = await runAnalysis(query, mode, {
      onStageUpdate: (stageNumber, status, durationMs) => {
        setStages((prev) =>
          prev.map((s) => (s.number === stageNumber ? { ...s, status, durationMs: durationMs ?? s.durationMs } : s))
        );
      },
      onResult: (res) => {
        setResult(res);
        setIsAnalyzing(false);
        setTotalTimeMs(res.processing_time_ms ?? performance.now() - startRef.current);
        setStages((prev) => prev.map((s) => ({ ...s, status: "completed" as StageStatus })));
      },
      onError: (message) => {
        setError(message);
        setIsAnalyzing(false);
      },
    });
  };

  return (
    <div className="flex min-h-screen bg-bg text-white">
      <Sidebar />

      <main className="flex-1 px-8 py-6 max-w-[1600px]">
        <TopBar />

        {error && (
          <div className="mb-5 px-4 py-3 rounded-xl bg-bad-soft border border-bad/40 text-bad text-[13px]">{error}</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_1.05fr_340px] gap-6 items-start">
          {/* Column 1 */}
          <div className="flex flex-col gap-6 min-w-0">
            <InputPanel
              mode={mode}
              onModeChange={setMode}
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleSubmit}
              isAnalyzing={isAnalyzing}
            />
            <ExecutionPipeline stages={stages} isLive={isAnalyzing} totalTimeMs={totalTimeMs} />
          </div>

          {/* Column 2 */}
          <div className="flex flex-col gap-6 min-w-0">
            <ModelInfoPanel jobId={result?.job_id} />
            <ResultTabs result={result} placeholderQuery={query} />
          </div>

          {/* Column 3 */}
          <div className="flex flex-col gap-6 min-w-0">
            <OverallResultPanel result={result} isAnalyzing={isAnalyzing} />
            <RetrievedEvidencePanel result={result} />
            <RequestDetailsPanel result={result} mode={mode} />
          </div>
        </div>

        <div className="mt-6 flex items-center gap-2 px-4 py-3 rounded-xl bg-accent-blue/10 border border-accent-blue/20 text-[13px] text-white/70">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-blue shrink-0" />
          Each step is processed in real-time. You can track the progress of the analysis pipeline on the left.
        </div>
      </main>
    </div>
  );
}
