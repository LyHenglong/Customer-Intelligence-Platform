import type { ReactNode } from "react";

/** The reference's page heading: small eyebrow, large bold title, one line
 *  of context, and an optional control docked right. Shared so every page
 *  gets the same type scale instead of each re-deciding it. */
export default function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-[13px]" style={{ color: "var(--text-secondary)" }}>
            {eyebrow}
          </p>
        )}
        <h1 className="mt-0.5 text-[26px] font-bold leading-tight tracking-tight" style={{ color: "var(--text-primary)" }}>
          {title}
        </h1>
        {description && (
          <p className="mt-1 text-[13px]" style={{ color: "var(--text-secondary)" }}>
            {description}
          </p>
        )}
      </div>
      {action}
    </div>
  );
}
