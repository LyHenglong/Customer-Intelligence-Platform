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
    <label className="flex items-center gap-3 text-sm">
      <span style={{ color: "var(--text-secondary)" }}>Churn probability threshold</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-40 accent-[var(--series-1)]"
      />
      <span className="w-12 tabular-nums font-medium" style={{ color: "var(--text-primary)" }}>
        {value.toFixed(2)}
      </span>
    </label>
  );
}
