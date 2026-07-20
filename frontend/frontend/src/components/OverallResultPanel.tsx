import { HeartPulse, AlertTriangle, CheckCircle2 } from "lucide-react";
import { AnalysisResult } from "../types";
import CircularGauge from "./CircularGauge";

interface OverallResultPanelProps {
  result: AnalysisResult | null;
  isAnalyzing: boolean;
}

export default function OverallResultPanel({ result, isAnalyzing }: OverallResultPanelProps) {
  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const isHallucination = result?.hallucination ?? false;

  return (
    <div className="panel p-5">
      <div className="flex items-center gap-2 mb-4">
        <HeartPulse className="w-4 h-4 text-accent-purpleLight" />
        <h2 className="text-white font-semibold text-[15px]">Overall Result</h2>
      </div>

      {!result ? (
        <div className="py-8 text-center">
          <p className="text-muted text-[13px]">{isAnalyzing ? "Analyzing..." : "Awaiting analysis"}</p>
        </div>
      ) : (
        <>
          <div
            className={`flex items-center gap-2 rounded-xl px-3.5 py-2.5 mb-5 border ${
              isHallucination ? "bg-bad-soft border-bad/40 text-bad" : "bg-good-soft border-good/40 text-good"
            }`}
          >
            {isHallucination ? <AlertTriangle className="w-4 h-4 shrink-0" /> : <CheckCircle2 className="w-4 h-4 shrink-0" />}
            <span className="text-[12.5px] font-bold tracking-wide">
              {isHallucination ? "HALLUCINATION DETECTED" : "APPEARS TRUTHFUL"}
            </span>
          </div>

          <div className="text-center mb-2">
            <span className="text-muted text-[13px]">Confidence Score</span>
          </div>
          <div className="flex justify-center mb-3">
            <CircularGauge
              value={confidencePct}
              size={140}
              strokeWidth={11}
              color={isHallucination ? "#f2555c" : "#22c55e"}
              sublabel={confidencePct >= 66 ? "High Confidence" : confidencePct >= 33 ? "Medium Confidence" : "Low Confidence"}
              sublabelColor={isHallucination ? "#f2555c" : "#22c55e"}
            />
          </div>

          <div className="mt-4">
            <div className="h-2 rounded-full overflow-hidden bg-white/5 relative">
              <div className="absolute inset-0 bg-gradient-to-r from-good via-warn to-bad" />
              <div
                className="absolute top-0 bottom-0 w-[3px] bg-white rounded-full shadow"
                style={{ left: `calc(${confidencePct}% - 1.5px)` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-muted mt-1.5">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
