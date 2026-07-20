import { Bot, MessageSquare, Play, Loader2 } from "lucide-react";
import { InputMode } from "../types";

const TABS: { key: InputMode; label: string }[] = [
  { key: "text", label: "Text" },
  { key: "image", label: "Image" },
  { key: "both", label: "Text + Image" },
];

const MAX_CHARS = 2000;

interface InputPanelProps {
  mode: InputMode;
  onModeChange: (mode: InputMode) => void;
  query: string;
  onQueryChange: (q: string) => void;
  onSubmit: () => void;
  isAnalyzing: boolean;
}

export default function InputPanel({ mode, onModeChange, query, onQueryChange, onSubmit, isAnalyzing }: InputPanelProps) {
  return (
    <div className="panel p-5">
      <div className="flex items-center gap-1 mb-5">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => onModeChange(tab.key)}
            className={`px-4 py-1.5 rounded-lg text-[13.5px] font-semibold transition-colors ${
              mode === tab.key ? "text-accent-purpleLight border-b-2 border-accent-purple" : "text-muted hover:text-white"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex gap-4">
        <div className="flex-1">
          <label className="text-[13px] text-white/80 font-medium mb-2 block">Enter your prompt</label>
          <div className="relative">
            <textarea
              value={query}
              maxLength={MAX_CHARS}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder="Who invented the telephone?"
              rows={4}
              className="w-full resize-none bg-bg-soft border border-border-subtle rounded-xl px-4 py-3 text-[14px] text-white placeholder:text-muted/60 focus:outline-none focus:ring-2 focus:ring-accent-purple/40 focus:border-accent-purple/50"
            />
            <span className="absolute bottom-2.5 right-3 text-[11px] text-muted">
              {query.length}/{MAX_CHARS}
            </span>
          </div>

          <button
            onClick={onSubmit}
            disabled={!query.trim() || isAnalyzing}
            className="mt-4 flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent-purple hover:bg-accent-purpleLight disabled:opacity-40 disabled:cursor-not-allowed text-white text-[13.5px] font-semibold transition-colors"
          >
            {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {isAnalyzing ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        <div className="hidden sm:flex w-28 shrink-0 items-end justify-center pb-1">
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-accent-purple/25 to-accent-blue/10 border border-accent-purple/30 flex items-center justify-center">
              <Bot className="w-10 h-10 text-accent-purpleLight" />
            </div>
            <div className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-accent-blue/20 border border-accent-blue/40 flex items-center justify-center">
              <MessageSquare className="w-3.5 h-3.5 text-accent-blue" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
