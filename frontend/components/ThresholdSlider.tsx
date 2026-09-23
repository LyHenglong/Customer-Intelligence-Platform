export default function ThresholdSlider({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label
      className="flex items-center gap-3 rounded-lg border px-3 py-2 text-[12.5px]"
      style={{
        borderColor: "var(--border)",
        background: "var(--surface-card)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      <span className="whitespace-nowrap font-medium" style={{ color: "var(--text-secondary)" }}>
        Churn probability threshold
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-36 accent-[var(--brand)]"
      />
      <span
        className="w-11 rounded-md px-1.5 py-0.5 text-center text-[12px] font-semibold tabular-nums"
        style={{ background: "var(--brand-tint)", color: "var(--brand-strong)" }}
      >
        {value.toFixed(2)}
      </span>
    </label>
  );
}
