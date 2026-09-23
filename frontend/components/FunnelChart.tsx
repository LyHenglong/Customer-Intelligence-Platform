export interface FunnelStage {
  label: string;
  value: number;
}

/**
 * Stacked tapering bands, as in the reference's conversion funnel.
 *
 * Colour here encodes magnitude, not identity, so it is a single hue
 * stepped light -> dark rather than the categorical series - a rainbow
 * funnel would imply the stages are unrelated categories. Each band is
 * labelled with its count and share, so the ramp is never the only thing
 * carrying the value.
 */
export default function FunnelChart({
  stages,
  valueFormatter = (v: number) => v.toLocaleString(),
}: {
  stages: FunnelStage[];
  valueFormatter?: (value: number) => string;
}) {
  if (stages.length === 0) return null;
  const top = stages[0].value || 1;

  return (
    <div className="flex flex-col gap-1.5">
      {stages.map((stage, i) => {
        // Width is the honest ratio to the first stage; a floor of 22%
        // keeps a tiny final stage still readable rather than collapsing
        // it to a sliver with unreadable text.
        const ratio = stage.value / top;
        const width = Math.max(ratio, 0.22) * 100;
        const shade = 0.34 + (i / Math.max(stages.length - 1, 1)) * 0.56;

        return (
          <div key={stage.label} className="flex items-center gap-3">
            <div className="flex min-w-0 flex-1 justify-center">
              <div
                className="flex items-center justify-center rounded-md px-3 py-2.5 text-[12px] font-semibold text-white transition-all"
                style={{
                  width: `${width}%`,
                  background: `color-mix(in srgb, var(--series-1) ${shade * 100}%, var(--surface-card))`,
                }}
                title={`${stage.label}: ${valueFormatter(stage.value)}`}
              >
                <span className="truncate">{stage.label.replace(/_/g, " ")}</span>
              </div>
            </div>
            <div className="flex w-32 shrink-0 items-baseline justify-end gap-2">
              <span className="tabular-nums text-[12.5px] font-semibold" style={{ color: "var(--text-primary)" }}>
                {valueFormatter(stage.value)}
              </span>
              <span className="tabular-nums text-[11px]" style={{ color: "var(--text-muted)" }}>
                {(ratio * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
