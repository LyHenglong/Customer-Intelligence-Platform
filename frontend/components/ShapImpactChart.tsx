import type { ShapFactor } from "@/lib/types";

function ImpactGroup({
  title,
  items,
  color,
  maxAbs,
}: {
  title: string;
  items: ShapFactor[];
  color: string;
  maxAbs: number;
}) {
  return (
    <div className="min-w-0 flex-1">
      <h3 className="mb-2.5 text-[12px] font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {title}
      </h3>
      {items.length === 0 ? (
        <p className="text-[12.5px]" style={{ color: "var(--text-muted)" }}>None identified.</p>
      ) : (
        <ul className="space-y-2.5">
          {items.map((f) => (
            <li key={f.feature}>
              <div className="mb-1 flex items-center justify-between gap-2 text-[12.5px]">
                <span style={{ color: "var(--text-secondary)" }}>{f.feature.replaceAll("_", " ")}</span>
                <span className="shrink-0 tabular-nums font-medium" style={{ color: "var(--text-primary)" }}>
                  {Math.abs(f.shap_value).toFixed(3)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--gridline)" }}>
                <div
                  className="h-full rounded-full"
                  style={{ width: `${(Math.abs(f.shap_value) / maxAbs) * 100}%`, background: color }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Full-size explainable-AI view for the customer profile page: SHAP
 *  factors split into two grouped, magnitude-ranked bar lists so a
 *  business user can see at a glance which factors push risk up versus
 *  down, without reading a legend. `ShapFactorList` stays the compact
 *  single-line version for dense table cells (at-risk, recommendations) -
 *  this is the expanded version for a page that has room for it. */
export default function ShapImpactChart({ factors }: { factors: ShapFactor[] }) {
  if (factors.length === 0) {
    return <p className="text-sm" style={{ color: "var(--text-muted)" }}>No risk factors available for this customer.</p>;
  }

  const increasing = factors
    .filter((f) => f.direction.toLowerCase().includes("increase"))
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
  const decreasing = factors
    .filter((f) => !f.direction.toLowerCase().includes("increase"))
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shap_value)), 1e-9);

  return (
    <div className="flex flex-col gap-6 sm:flex-row">
      <ImpactGroup title="Factors increasing risk" items={increasing} color="var(--status-critical)" maxAbs={maxAbs} />
      <ImpactGroup title="Factors reducing risk" items={decreasing} color="var(--status-good)" maxAbs={maxAbs} />
    </div>
  );
}
