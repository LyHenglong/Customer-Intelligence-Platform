export type RiskTier = "high" | "medium" | "low";

/** Fixed probability cut points, not tied to the model's decision
 *  threshold - the threshold says "act on this customer", these tiers say
 *  "how urgently", and the two questions have different natural cut
 *  points. Stated here rather than hidden so the mapping is auditable. */
export function riskTier(churnProbability: number): RiskTier {
  if (churnProbability >= 0.7) return "high";
  if (churnProbability >= 0.4) return "medium";
  return "low";
}

const TIER_STYLE: Record<RiskTier, { label: string; fg: string; bg: string }> = {
  high: { label: "High risk", fg: "var(--status-critical)", bg: "color-mix(in srgb, var(--status-critical) 14%, transparent)" },
  medium: { label: "Medium risk", fg: "var(--status-warning)", bg: "color-mix(in srgb, var(--status-warning) 18%, transparent)" },
  low: { label: "Low risk", fg: "var(--status-good)", bg: "color-mix(in srgb, var(--status-good) 14%, transparent)" },
};

export default function RiskBadge({ churnProbability, tier }: { churnProbability?: number; tier?: RiskTier }) {
  const resolved = tier ?? (churnProbability !== undefined ? riskTier(churnProbability) : "low");
  const style = TIER_STYLE[resolved];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11.5px] font-semibold whitespace-nowrap"
      style={{ background: style.bg, color: style.fg }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: style.fg }} aria-hidden="true" />
      {style.label}
    </span>
  );
}
