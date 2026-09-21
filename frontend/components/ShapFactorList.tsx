import type { ShapFactor } from "@/lib/types";

export default function ShapFactorList({ factors }: { factors: ShapFactor[] }) {
  if (factors.length === 0) {
    return <span className="text-xs" style={{ color: "var(--text-muted)" }}>No risk factors available</span>;
  }
  return (
    <ul className="space-y-1">
      {factors.map((f) => {
        const increases = f.direction.toLowerCase().includes("increase");
        return (
          <li key={f.feature} className="flex items-center gap-1.5 text-xs">
            <span aria-hidden style={{ color: increases ? "var(--status-critical)" : "var(--status-good)" }}>
              {increases ? "▲" : "▼"}
            </span>
            <span style={{ color: "var(--text-secondary)" }}>{f.feature.replaceAll("_", " ")}</span>
          </li>
        );
      })}
    </ul>
  );
}
