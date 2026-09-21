"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface BarDatum {
  label: string;
  value: number;
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
            <YAxis type="category" dataKey="label" width={140} tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
          </>
        ) : (
          <>
            <XAxis dataKey="label" tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
            <YAxis tickFormatter={valueFormatter} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
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
        <Bar dataKey="value" fill="var(--series-1)" radius={layout === "horizontal" ? [0, 4, 4, 0] : [4, 4, 0, 0]} maxBarSize={48} />
      </BarChart>
    </ResponsiveContainer>
  );
}
