export default function KpiCard({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <div
      className="rounded-lg border p-4"
      style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}
    >
      <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold" style={{ color: "var(--text-primary)" }}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
          {sublabel}
        </div>
      )}
    </div>
  );
}
