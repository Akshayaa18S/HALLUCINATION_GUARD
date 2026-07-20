import { Download, Sun, Bell } from "lucide-react";

export default function TopBar() {
  return (
    <div className="flex items-center justify-between mb-6">
      <div>
        <h1 className="text-white text-2xl font-bold">New Analysis</h1>
        <p className="text-muted text-[13.5px] mt-0.5">Real-time hallucination detection in progress</p>
      </div>
      <div className="flex items-center gap-3">
        <button className="flex items-center gap-2 px-4 py-2 rounded-xl border border-border-subtle text-[13.5px] text-white/90 hover:bg-white/5 transition-colors">
          <Download className="w-4 h-4" />
          Export Report
        </button>
        <button className="w-9 h-9 rounded-xl border border-border-subtle flex items-center justify-center text-muted hover:text-white hover:bg-white/5 transition-colors">
          <Sun className="w-4 h-4" />
        </button>
        <button className="w-9 h-9 rounded-xl bg-accent-purple/15 border border-accent-purple/30 flex items-center justify-center text-accent-purpleLight">
          <Bell className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
