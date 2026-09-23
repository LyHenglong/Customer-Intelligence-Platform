import type { ReactNode } from "react";

/** Panel with the reference's header row: title on the left, an optional
 *  control or link on the right. Every dashboard panel uses this so the
 *  padding, radius and header rhythm stay identical across pages. */
export default function Card({
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`rounded-xl border ${className}`}
      style={{
        borderColor: "var(--border)",
        background: "var(--surface-card)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      {(title || action) && (
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-3">
          {typeof title === "string" ? (
            <h2 className="text-[13.5px] font-semibold" style={{ color: "var(--text-primary)" }}>
              {title}
            </h2>
          ) : (
            title
          )}
          {action}
        </div>
      )}
      <div className={`px-5 pb-5 ${title || action ? "" : "pt-5"} ${bodyClassName}`}>{children}</div>
    </section>
  );
}
