import { AnalysisResult, InputMode } from "../types";

interface RequestDetailsPanelProps {
  result: AnalysisResult | null;
  mode: InputMode;
}

function formatElapsed(ms?: number): string {
  if (!ms) return "—";
  const totalSeconds = ms / 1000;
  const mm = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const ss = (totalSeconds % 60).toFixed(0).padStart(2, "0");
  return `00:${mm}:${ss}`;
}

export default function RequestDetailsPanel({ result, mode }: RequestDetailsPanelProps) {
  const rows = [
    { label: "Input Type", value: mode === "text" ? "Text" : mode === "image" ? "Image" : "Text + Image" },
    { label: "Tokens (Input)", value: result?.tokens_input?.toString() || "—" },
    { label: "Tokens (Output)", value: result?.tokens_output?.toString() || "—" },
    { label: "Analysis Time", value: formatElapsed(result?.processing_time_ms) },
    { label: "Request Time", value: result?.created_at ? new Date(result.created_at).toLocaleString() : new Date().toLocaleString() },
  ];

  return (
    <div className="panel p-5">
      <h2 className="text-white font-semibold text-[15px] mb-4">Request Details</h2>
      <div className="space-y-3">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between">
            <span className="text-muted text-[13px]">{row.label}</span>
            <span className="text-white/90 text-[13px] font-medium">{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
