"use client";

import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface BarDatum {
  label: string;
  value: number;
}

/** Warehouse keys arrive as `established_6_24mo`. Underscores render as one
 *  long unbreakable token, which is what made Recharts silently drop the
 *  middle tick on the tenure chart - three bars, two labels. */
const prettify = (label: string) => label.replace(/_/g, " ");

/** Wraps a category tick over up to two lines.
 *
 *  Long warehouse keys ("established 6 24mo") either get silently dropped
 *  by Recharts' collision thinning, or - once interval={0} forces them all
 *  to render - overlap their neighbours into unreadable mush. Wrapping
 *  keeps every category labelled and legible without angling the text. */
function WrappedTick({ x, y, payload }: { x?: number; y?: number; payload?: { value?: string | number } }) {
  const words = prettify(String(payload?.value ?? "")).split(" ");
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > 10 && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
    if (lines.length === 1 && current.length > 10) break;
  }
  if (current) lines.push(current);

  return (
    <g transform={`translate(${x ?? 0},${y ?? 0})`}>
      {lines.slice(0, 2).map((line, i) => (
        <text key={i} x={0} y={0} dy={12 + i * 11} textAnchor="middle" fill="var(--text-secondary)" fontSize={11}>
          {line}
        </text>
      ))}
    </g>
  );
}

export default function ChurnBarChart({
  data,
  valueFormatter = (v: number) => v.toFixed(2),
  height = 260,
  layout = "vertical",
}: {
  data: BarDatum[];
  valueFormatter?: (value: number) => string;
  height?: number;
  layout?: "vertical" | "horizontal"; // "vertical" = bars stand up (categories on x); "horizontal" = bars lie flat
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={layout === "horizontal" ? "vertical" : "horizontal"}
        margin={{ top: 8, right: 12, bottom: 8, left: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--gridline)" horizontal={layout !== "horizontal"} vertical={layout === "horizontal"} />
        {layout === "horizontal" ? (
          <>
            <XAxis type="number" tickFormatter={valueFormatter} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
            <YAxis type="category" dataKey="label" width={140} tickFormatter={prettify} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
          </>
        ) : (
          <>
            {/* interval={0} forces every category to render rather than
                letting Recharts thin them out when they collide. */}
            <XAxis
              dataKey="label"
              interval={0}
              height={40}
              tick={<WrappedTick />}
              axisLine={{ stroke: "var(--axis)" }}
              tickLine={false}
            />
            <YAxis tickFormatter={valueFormatter} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} width={44} />
          </>
        )}
        <Tooltip
          formatter={(value) => valueFormatter(Number(value))}
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            fontSize: 12,
            color: "var(--text-primary)",
          }}
          cursor={{ fill: "var(--gridline)", opacity: 0.4 }}
        />
        {/* Value labels are not decoration: the palette validator flags
            --series-1 at 2.74:1 against the card, which is only acceptable
            with secondary encoding. These labels are that encoding, so the
            bars stay readable without relying on fill contrast (and match
            the reference, which labels every bar end). */}
        <Bar dataKey="value" fill="var(--series-1)" radius={layout === "horizontal" ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={48}>
          <LabelList
            dataKey="value"
            position={layout === "horizontal" ? "right" : "top"}
            formatter={(value: unknown) => valueFormatter(Number(value))}
            style={{ fill: "var(--text-secondary)", fontSize: 11, fontWeight: 600 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
