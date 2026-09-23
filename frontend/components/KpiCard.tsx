import type { ReactNode } from "react";

export type KpiTone = "brand" | "info" | "warning" | "violet" | "danger";

const TONES: Record<KpiTone, { fg: string; bg: string }> = {
  brand: { fg: "var(--series-1)", bg: "color-mix(in srgb, var(--series-1) 12%, transparent)" },
  info: { fg: "var(--series-2)", bg: "color-mix(in srgb, var(--series-2) 12%, transparent)" },
  warning: { fg: "var(--series-3)", bg: "color-mix(in srgb, var(--series-3) 16%, transparent)" },
  violet: { fg: "var(--series-4)", bg: "color-mix(in srgb, var(--series-4) 12%, transparent)" },
  danger: { fg: "var(--series-5)", bg: "color-mix(in srgb, var(--series-5) 12%, transparent)" },
};

/**
 * The reference pairs each KPI with a tinted icon chip and a
 * "+18.3% vs last month" delta. `trend` is optional and never synthesised:
 * this platform has no month-over-month baseline, so a KPI without a real
 * comparison simply renders without one rather than showing an invented
 * percentage.
 */
export default function KpiCard({
  label,
  value,
  sublabel,
  icon,
  tone = "brand",
  trend,
}: {
  label: string;
  value: string;
  sublabel?: string;
  icon?: ReactNode;
  tone?: KpiTone;
  trend?: { value: string; direction: "up" | "down"; label?: string; good?: boolean };
}) {
  const toneColors = TONES[tone];
  // Direction is which way the number moved; `good` is whether that is
  // welcome. They differ constantly here - rising churn points up and is
  // bad - so colour follows `good` and the arrow follows `direction`.
  const trendGood = trend?.good ?? trend?.direction === "up";

  return (
    // h-full + flex-col so a card whose label wraps, or which has no
    // sublabel, still matches its neighbours' height in the KPI row.
    <div
      className="flex h-full flex-col rounded-xl border p-4"
      style={{
        borderColor: "var(--border)",
        background: "var(--surface-card)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div className="flex items-start gap-3">
        {icon && (
          <span
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
            style={{ background: toneColors.bg, color: toneColors.fg }}
            aria-hidden="true"
          >
            {icon}
          </span>
        )}
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
            {label}
          </div>
          <div
            className="mt-0.5 truncate text-[22px] font-bold leading-tight tracking-tight"
            style={{ color: "var(--text-primary)" }}
            title={value}
          >
            {value}
          </div>
        </div>
      </div>

      {(trend || sublabel) && (
        <div className="mt-auto flex flex-wrap items-center gap-x-2 gap-y-1 pt-2.5">
          {trend && (
            <span
              className="inline-flex items-center gap-1 text-[12px] font-semibold"
              style={{ color: trendGood ? "var(--status-good)" : "var(--status-critical)" }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="h-3 w-3" aria-hidden="true">
                {trend.direction === "up" ? <><path d="M6 18 18 6" /><path d="M9 6h9v9" /></> : <><path d="M6 6l12 12" /><path d="M18 9v9H9" /></>}
              </svg>
              {trend.value}
            </span>
          )}
          {(trend?.label || sublabel) && (
            <span className="text-[11.5px]" style={{ color: "var(--text-secondary)" }}>
              {trend?.label ?? sublabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
