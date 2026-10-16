"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

/* Icons are inline SVG rather than an icon package: seven glyphs did not
   justify another dependency in the browser bundle. All share a 24x24 box
   and currentColor so the active/inactive colour is driven by the link. */
const icon = (path: ReactNode) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-[18px] w-[18px] shrink-0"
    aria-hidden="true"
  >
    {path}
  </svg>
);

const NAV = [
  {
    href: "/overview",
    label: "Overview",
    icon: icon(<><path d="M3 10.5 12 3l9 7.5" /><path d="M5 9.5V21h14V9.5" /></>),
  },
  {
    href: "/at-risk",
    label: "At-Risk Customers",
    icon: icon(<><path d="M12 3.5 2.8 19.5h18.4L12 3.5Z" /><path d="M12 10v4" /><path d="M12 17.2h.01" /></>),
  },
  {
    href: "/customers",
    label: "Customers",
    icon: icon(<><circle cx="9" cy="8" r="3.2" /><path d="M2.5 20a6.5 6.5 0 0 1 13 0" /><path d="M16.5 5.6a3.2 3.2 0 0 1 0 6.3" /><path d="M18 14.3a6.5 6.5 0 0 1 3.5 5.7" /></>),
  },
  {
    href: "/segments",
    label: "Segments",
    icon: icon(<><circle cx="12" cy="12" r="8.5" /><path d="M12 3.5v8.5h8.5" /></>),
  },
  {
    href: "/insights",
    label: "Insights",
    icon: icon(<><path d="M12 2.5a5.5 5.5 0 0 0-3.2 9.98c.5.36.7.86.7 1.3v.72h5v-.72c0-.44.2-.94.7-1.3A5.5 5.5 0 0 0 12 2.5Z" /><path d="M10 18.5h4" /><path d="M10.5 21h3" /></>),
  },
  {
    href: "/recommendations",
    label: "Recommendations",
    icon: icon(<><path d="M9 11.5 11 13.5 15.5 9" /><rect x="3.5" y="3.5" width="17" height="17" rx="3.5" /></>),
  },
  {
    href: "/model-performance",
    label: "Model Performance",
    icon: icon(<><path d="M3 20h18" /><path d="M6 20v-6" /><path d="M11 20V7" /><path d="M16 20v-9" /><path d="M21 20V4" /></>),
  },
  {
    href: "/pipeline-status",
    label: "Pipeline Status",
    icon: icon(<><path d="M4 7h10" /><circle cx="17.5" cy="7" r="2.5" /><path d="M20 17H10" /><circle cx="6.5" cy="17" r="2.5" /></>),
  },
  {
    href: "/assistant",
    label: "AI Assistant",
    icon: icon(<><path d="M12 3v2.2" /><rect x="4" y="5.2" width="16" height="13" rx="3.2" /><path d="M9.2 11v1.6" /><path d="M14.8 11v1.6" /><path d="M9.5 21h5" /></>),
  },
];

function isActive(pathname: string | null, href: string) {
  if (!pathname) return false;
  // /customers must not light up while on /customers/CUST123's detail page's
  // *sibling* routes, but should stay lit on the detail page itself.
  return pathname === href || pathname.startsWith(href + "/");
}

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="hidden w-60 shrink-0 flex-col border-r lg:flex"
      style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}
    >
      <div className="flex h-16 items-center gap-2.5 px-5">
        <span
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
          style={{ background: "var(--brand)" }}
          aria-hidden="true"
        >
          {/* Retention Loop: an open ring whose gap is closed by a customer
              dot - a customer kept in the loop. Echoes ChurnGauge's ring. */}
          <svg viewBox="0 0 48 48" fill="none" className="h-8 w-8">
            <path d="M34.63 21.15A11 11 0 1 1 21.15 13.37" stroke="white" strokeWidth="4.5" strokeLinecap="round" />
            <circle cx="29.5" cy="14.47" r="2.8" fill="white" />
          </svg>
        </span>
        {/* Stacked: "Customer Intelligence" on one line doesn't fit beside
            the mark in a w-60 sidebar at a legible size. */}
        <span className="flex flex-col text-[14px] font-bold leading-[1.15] tracking-tight">
          <span style={{ color: "var(--text-primary)" }}>Customer</span>
          <span style={{ color: "var(--brand-strong)" }}>Intelligence</span>
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3 py-2">
        {NAV.map((item) => {
          const active = isActive(pathname, item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13.5px] font-medium transition-colors"
              style={{
                color: active ? "var(--brand-strong)" : "var(--text-secondary)",
                background: active ? "var(--brand-tint)" : "transparent",
              }}
            >
              {item.icon}
              <span className="truncate">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="px-3 pb-4">
        <div
          className="rounded-lg p-3 text-[11.5px] leading-relaxed"
          style={{ background: "var(--surface-sunken)", color: "var(--text-muted)" }}
        >
          <span className="font-semibold" style={{ color: "var(--text-secondary)" }}>
            Synthetic data
          </span>
          <br />
          Figures come from a generated 1M-row dataset, not a real market.
        </div>
      </div>
    </aside>
  );
}
