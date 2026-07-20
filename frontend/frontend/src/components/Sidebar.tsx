import {
  PlusCircle,
  LayoutDashboard,
  FileText,
  ImageIcon,
  History,
  ClipboardList,
  Database,
  Settings,
  HelpCircle,
  BookOpen,
  ShieldCheck,
  ChevronDown,
  Moon,
} from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = [
  { icon: PlusCircle, label: "New Analysis", active: true },
  { icon: LayoutDashboard, label: "Dashboard" },
  { icon: FileText, label: "Text Analysis" },
  { icon: ImageIcon, label: "Image Analysis" },
  { icon: History, label: "History" },
  { icon: ClipboardList, label: "Reports" },
  { icon: Database, label: "Datasets" },
  { icon: Settings, label: "Settings" },
  { icon: BookOpen, label: "API Docs" },
  { icon: HelpCircle, label: "Help & Support" },
];

export default function Sidebar() {
  const [darkMode, setDarkMode] = useState(true);

  return (
    <aside className="w-[260px] shrink-0 bg-bg-sidebar border-r border-border-subtle flex flex-col h-screen sticky top-0">
      <div className="flex items-center gap-3 px-6 py-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-purple to-accent-blue flex items-center justify-center shadow-lg shadow-accent-purple/20">
          <ShieldCheck className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="text-white font-bold text-[15px] leading-tight">Hallucination Guard</div>
          <div className="text-muted text-xs leading-tight">AI Truth Verifier</div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map(({ icon: Icon, label, active }) => (
          <button
            key={label}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13.5px] font-medium transition-colors ${
              active
                ? "bg-accent-purple/15 text-accent-purpleLight border border-accent-purple/30"
                : "text-muted hover:text-white hover:bg-white/5 border border-transparent"
            }`}
          >
            <Icon className="w-[18px] h-[18px]" />
            {label}
          </button>
        ))}
      </nav>

      <div className="px-3 pb-4 space-y-3">
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center text-white text-xs font-bold">
            A
          </div>
          <div className="text-left flex-1">
            <div className="text-white text-[13px] font-semibold leading-tight">Anubhav</div>
            <div className="text-muted text-[11px] leading-tight">Premium User</div>
          </div>
          <ChevronDown className="w-4 h-4 text-muted" />
        </button>

        <div className="flex items-center justify-between px-3 py-2.5 rounded-xl bg-white/5">
          <div className="flex items-center gap-2 text-muted text-[13px] font-medium">
            <Moon className="w-4 h-4" />
            Dark Mode
          </div>
          <button
            onClick={() => setDarkMode((v) => !v)}
            className={`w-9 h-5 rounded-full relative transition-colors ${
              darkMode ? "bg-accent-purple" : "bg-white/20"
            }`}
          >
            <span
              className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
                darkMode ? "left-[18px]" : "left-0.5"
              }`}
            />
          </button>
        </div>
      </div>
    </aside>
  );
}
