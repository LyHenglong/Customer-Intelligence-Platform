export default function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="flex items-center justify-center rounded-lg border p-12" style={{ borderColor: "var(--border)" }}>
      <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
        {label}
      </span>
    </div>
  );
}
