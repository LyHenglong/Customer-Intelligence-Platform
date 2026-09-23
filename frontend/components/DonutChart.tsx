"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

export interface DonutDatum {
  label: string;
  value: number;
}

// Fixed order, never cycled - slice colour follows the entity, so filtering
// the data cannot repaint the survivors. Only five slots exist because
// that is what the palette validator passed; a sixth category folds into
// "Other" upstream rather than inventing a hue.
const SLICE_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export default function DonutChart({
  data,
  centerValue,
  centerLabel,
  valueFormatter = (v: number) => v.toLocaleString(),
  height = 240,
}: {
  data: DonutDatum[];
  centerValue: string;
  centerLabel: string;
  valueFormatter?: (value: number) => string;
  height?: number;
}) {
  const total = data.reduce((sum, d) => sum + d.value, 0) || 1;

  return (
    <div className="flex flex-col items-center gap-4 sm:flex-row">
      <div className="relative shrink-0" style={{ width: height, height }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="label"
              innerRadius="62%"
              outerRadius="88%"
              paddingAngle={2}
              stroke="var(--surface-card)"
              strokeWidth={2}
              isAnimationActive={false}
            >
              {data.map((d, i) => (
                <Cell key={d.label} fill={SLICE_COLORS[i % SLICE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value, name) => [valueFormatter(Number(value)), String(name).replace(/_/g, " ")]}
              contentStyle={{
                background: "var(--surface-card)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                fontSize: 12,
                color: "var(--text-primary)",
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        {/* Hero number in the hole, as in the reference. pointer-events-none
            so it never blocks a slice's hover target. */}
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[22px] font-bold leading-none tracking-tight" style={{ color: "var(--text-primary)" }}>
            {centerValue}
          </span>
          <span className="mt-1 text-[11px]" style={{ color: "var(--text-muted)" }}>
            {centerLabel}
          </span>
        </div>
      </div>

      {/* Legend carries the label and share as text, so identity is never
          colour-alone - the swatch is secondary. */}
      <ul className="flex w-full min-w-0 flex-col gap-2.5">
        {data.map((d, i) => (
          <li key={d.label} className="flex items-center gap-2.5 text-[12.5px]">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ background: SLICE_COLORS[i % SLICE_COLORS.length] }}
              aria-hidden="true"
            />
            <span className="min-w-0 flex-1 truncate" style={{ color: "var(--text-secondary)" }}>
              {d.label.replace(/_/g, " ")}
            </span>
            <span className="tabular-nums font-semibold" style={{ color: "var(--text-primary)" }}>
              {((d.value / total) * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
