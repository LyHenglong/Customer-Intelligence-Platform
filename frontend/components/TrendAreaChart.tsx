"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface TrendPoint {
  label: string;
  value: number;
}

/** Single-series line with a gradient wash under it, as in the reference's
 *  acquisition panel. One series, so no legend - the card title names it. */
export default function TrendAreaChart({
  data,
  valueFormatter = (v: number) => v.toLocaleString(),
  height = 240,
  tooltipLabel,
}: {
  data: TrendPoint[];
  valueFormatter?: (value: number) => string;
  height?: number;
  tooltipLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 4 }}>
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--series-1)" stopOpacity={0.28} />
            <stop offset="100%" stopColor="var(--series-1)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
          minTickGap={16}
        />
        <YAxis
          tickFormatter={valueFormatter}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={48}
        />
        <Tooltip
          formatter={(value) => [valueFormatter(Number(value)), tooltipLabel ?? "Value"]}
          contentStyle={{
            background: "var(--surface-card)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            fontSize: 12,
            color: "var(--text-primary)",
          }}
          cursor={{ stroke: "var(--axis)", strokeDasharray: "3 3" }}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke="var(--series-1)"
          strokeWidth={2}
          fill="url(#trendFill)"
          dot={{ r: 3, fill: "var(--series-1)", strokeWidth: 0 }}
          activeDot={{ r: 5, stroke: "var(--surface-card)", strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
