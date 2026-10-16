import { riskTier } from "./RiskBadge";

const TIER_COLOR = {
  high: "var(--status-critical)",
  medium: "var(--status-warning)",
  low: "var(--status-good)",
} as const;

/** Single-value probability ring for the customer profile header. Distinct
 *  from `DonutChart`, which is a categorical composition chart - this is a
 *  gauge for exactly one number, so it carries its own simpler SVG rather
 *  than forcing a one-slice pie through the composition chart's API. */
export default function ChurnGauge({ probability, size = 132 }: { probability: number; size?: number }) {
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(1, Math.max(0, probability));
  const offset = circumference * (1 - clamped);
  const color = TIER_COLOR[riskTier(clamped)];

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--gridline)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[24px] font-bold leading-none tracking-tight" style={{ color: "var(--text-primary)" }}>
          {(clamped * 100).toFixed(0)}%
        </span>
        <span className="mt-1 text-[10.5px]" style={{ color: "var(--text-muted)" }}>
          churn probability
        </span>
      </div>
    </div>
  );
}
