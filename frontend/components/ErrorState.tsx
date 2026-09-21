export default function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="flex items-center gap-2 rounded-lg border p-4 text-sm"
      style={{ borderColor: "var(--status-critical)", color: "var(--status-critical)" }}
      role="alert"
    >
      <span aria-hidden>⚠</span>
      <span>{message}</span>
    </div>
  );
}
