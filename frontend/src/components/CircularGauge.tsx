interface CircularGaugeProps {
  value: number; // 0-100
  size?: number;
  strokeWidth?: number;
  color: string;
  trackColor?: string;
  label?: string;
  sublabel?: string;
  sublabelColor?: string;
}

export default function CircularGauge({
  value,
  size = 128,
  strokeWidth = 10,
  color,
  trackColor = "#1c2333",
  label,
  sublabel,
  sublabelColor = "#f2555c",
}: CircularGaugeProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(100, Math.max(0, value)) / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={radius} stroke={trackColor} strokeWidth={strokeWidth} fill="none" />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-white">{Math.round(value)}%</span>
        </div>
      </div>
      {label && <div className="mt-2 text-sm text-muted text-center">{label}</div>}
      {sublabel && (
        <div className="text-sm font-semibold text-center" style={{ color: sublabelColor }}>
          {sublabel}
        </div>
      )}
    </div>
  );
}
