import { ExternalLink, ChevronDown } from "lucide-react";
import { AnalysisResult } from "../types";

interface RetrievedEvidencePanelProps {
  result: AnalysisResult | null;
}

export default function RetrievedEvidencePanel({ result }: RetrievedEvidencePanelProps) {
  const evidence = result?.retrieved_evidence;
  const docs = evidence?.supporting_documents || [];
  const passages = evidence?.evidence || [];

  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-white font-semibold text-[15px]">Retrieved Evidence</h2>
        {docs.length > 0 && (
          <button className="flex items-center gap-1 text-[12px] text-muted border border-border-subtle rounded-lg px-2.5 py-1">
            {docs[0]}
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {!result ? (
        <p className="text-muted text-[13px]">Evidence will appear here once analysis completes.</p>
      ) : passages.length === 0 ? (
        <p className="text-muted text-[13px]">No supporting evidence retrieved for this claim.</p>
      ) : (
        <div className="space-y-4">
          {passages.map((p, i) => (
            <div key={i} className="bg-bg-soft border border-border-subtle rounded-xl p-4">
              <p className="text-[13.5px] text-white/85 leading-relaxed">&ldquo;{p}&rdquo;</p>
              {docs[i] && (
                <a href="#" className="mt-2 flex items-center gap-1 text-[12.5px] text-accent-blue hover:underline">
                  {docs[i]}
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
