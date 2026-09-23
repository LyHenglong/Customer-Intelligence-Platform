import type { ReactNode } from "react";

export interface FeedItem {
  id: string;
  badge: ReactNode;
  title: string;
  subtitle?: string;
  meta?: string;
  status?: "good" | "warning" | "critical" | "neutral";
}

const STATUS_COLOR: Record<NonNullable<FeedItem["status"]>, string> = {
  good: "var(--status-good)",
  warning: "var(--status-warning)",
  critical: "var(--status-critical)",
  neutral: "var(--text-muted)",
};

/**
 * The reference's bottom row is three feeds: tasks, messages and team
 * activity, each row an avatar plus a line of text.
 *
 * This keeps that shape but carries real platform events - ingested
 * batches, trained model versions, drift checks. The avatars are
 * deliberately not reproduced: they belong to named people in a CRM, and
 * this platform has no users, so any face or name here would be invented.
 * A compact badge (batch number, version, severity) does the same
 * scanning job honestly.
 */
export default function FeedList({ items, emptyMessage = "Nothing to show yet." }: { items: FeedItem[]; emptyMessage?: string }) {
  if (items.length === 0) {
    return (
      <p className="py-6 text-center text-[12.5px]" style={{ color: "var(--text-muted)" }}>
        {emptyMessage}
      </p>
    );
  }

  return (
    <ul className="flex flex-col">
      {items.map((item, i) => (
        <li
          key={item.id}
          className="flex items-center gap-3 py-2.5"
          style={i > 0 ? { borderTop: "1px solid var(--border)" } : undefined}
        >
          <span
            className="flex h-9 w-9 shrink-0 flex-col items-center justify-center rounded-lg text-[10px] font-bold leading-tight"
            style={{ background: "var(--surface-sunken)", color: "var(--text-secondary)" }}
            aria-hidden="true"
          >
            {item.badge}
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[12.5px] font-medium" style={{ color: "var(--text-primary)" }}>
              {item.title}
            </p>
            {item.subtitle && (
              <p className="truncate text-[11.5px]" style={{ color: "var(--text-muted)" }}>
                {item.subtitle}
              </p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {item.meta && (
              <span className="whitespace-nowrap text-[11px]" style={{ color: "var(--text-muted)" }}>
                {item.meta}
              </span>
            )}
            {item.status && (
              <span
                className="h-2 w-2 rounded-full"
                style={{ background: STATUS_COLOR[item.status] }}
                aria-hidden="true"
              />
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
