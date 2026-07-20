import { CheckCircle2, Loader2, Circle, Rocket } from "lucide-react";
import { StageState } from "../types";

interface ExecutionPipelineProps {
  stages: StageState[];
  isLive: boolean;
  totalTimeMs: number | null;
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const mm = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const ss = (totalSeconds % 60).toString().padStart(2, "0");
  const cs = Math.floor((ms % 1000) / 10)
    .toString()
    .padStart(2, "0");
  return `00:${mm !== "00" ? mm : ss}:${mm !== "00" ? ss : cs}`;
}

export default function ExecutionPipeline({ stages, isLive, totalTimeMs }: ExecutionPipelineProps) {
  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-white font-semibold text-[15px]">Execution Pipeline</h2>
        {isLive && (
          <div className="flex items-center gap-1.5 text-good text-[12px] font-medium">
            <span className="w-2 h-2 rounded-full bg-good pulse-dot" />
            Live
          </div>
        )}
      </div>

      <div className="relative">
        {stages.map((stage, idx) => {
          const isLast = idx === stages.length - 1;
          const isDone = stage.status === "completed";
          const isRunning = stage.status === "running";

          return (
            <div key={stage.number} className="relative flex gap-3 pb-5 last:pb-0">
              {!isLast && (
                <span
                  className={`absolute left-[13px] top-7 bottom-0 w-[2px] ${
                    isDone ? "bg-good/60" : "bg-border-subtle"
                  }`}
                />
              )}

              <div
                className={`relative z-10 w-7 h-7 rounded-full flex items-center justify-center shrink-0 border-2 text-[11px] font-bold ${
                  isDone
                    ? "bg-good/15 border-good text-good"
                    : isRunning
                    ? "bg-accent-purple/15 border-accent-purple text-accent-purpleLight"
                    : "bg-white/5 border-border-subtle text-muted"
                }`}
              >
                {idx === stages.length - 1 && isDone ? (
                  <Rocket className="w-3.5 h-3.5" />
                ) : isDone ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : isRunning ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  stage.number
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-white text-[13.5px] font-medium">
                    {stage.number}. {stage.name}
                  </span>
                  {stage.durationMs !== undefined && (
                    <span className="text-muted text-[11px] shrink-0">{formatTime(stage.durationMs)}</span>
                  )}
                </div>
                <div className="flex items-center justify-between gap-2 mt-0.5">
                  <span className="text-muted text-[12px] truncate">{stage.description}</span>
                  <span
                    className={`text-[11px] font-medium shrink-0 flex items-center gap-1 ${
                      isDone ? "text-good" : isRunning ? "text-accent-purpleLight" : "text-muted"
                    }`}
                  >
                    {isDone && <CheckCircle2 className="w-3 h-3" />}
                    {isRunning && <Loader2 className="w-3 h-3 animate-spin" />}
                    {stage.status === "pending" && <Circle className="w-3 h-3" />}
                    {isDone ? "Completed" : isRunning ? "Running" : "Pending"}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {totalTimeMs !== null && (
        <div className="mt-4 pt-4 border-t border-border-subtle text-[13px] text-white/80">
          Total Execution Time: <span className="text-accent-purpleLight font-semibold">{formatTime(totalTimeMs)}</span>
        </div>
      )}
    </div>
  );
}
