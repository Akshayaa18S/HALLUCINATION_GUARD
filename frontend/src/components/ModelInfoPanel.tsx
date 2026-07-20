interface ModelInfoPanelProps {
  jobId?: string;
}

export default function ModelInfoPanel({ jobId }: ModelInfoPanelProps) {
  const rows = [
    { label: "LLM Model", value: "Claude (Anthropic API)" },
    { label: "Detection Model", value: "MultiHaluDet (Ensemble)" },
    { label: "RAG Source", value: "Wikipedia + FEVER" },
    { label: "Language", value: "English" },
    { label: "Request ID", value: jobId || "—" },
  ];

  return (
    <div className="panel p-5">
      <h2 className="text-white font-semibold text-[15px] mb-4">Model Information</h2>
      <div className="space-y-3">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between">
            <span className="text-muted text-[13px]">{row.label}</span>
            <span className="text-white/90 text-[13px] font-medium truncate max-w-[55%] text-right">{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
