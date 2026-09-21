export default function PaginationControls({
  offset,
  limit,
  total,
  onChange,
}: {
  offset: number;
  limit: number;
  total: number;
  onChange: (offset: number) => void;
}) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));

  return (
    <div className="flex items-center justify-between text-sm" style={{ color: "var(--text-secondary)" }}>
      <span>
        Showing {total === 0 ? 0 : offset + 1}-{Math.min(offset + limit, total)} of {total.toLocaleString()}
      </span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={offset === 0}
          onClick={() => onChange(Math.max(0, offset - limit))}
          className="rounded-md border px-2.5 py-1 disabled:opacity-40"
          style={{ borderColor: "var(--border)" }}
        >
          Previous
        </button>
        <span className="tabular-nums">
          Page {page} / {totalPages}
        </span>
        <button
          type="button"
          disabled={offset + limit >= total}
          onClick={() => onChange(offset + limit)}
          className="rounded-md border px-2.5 py-1 disabled:opacity-40"
          style={{ borderColor: "var(--border)" }}
        >
          Next
        </button>
      </div>
    </div>
  );
}
