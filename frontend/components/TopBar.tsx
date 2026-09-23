"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

const NAV = [
  { href: "/overview", label: "Overview" },
  { href: "/at-risk", label: "At-Risk" },
  { href: "/customers", label: "Customers" },
  { href: "/segments", label: "Segments" },
  { href: "/model-performance", label: "Model" },
  { href: "/pipeline-status", label: "Pipeline" },
  { href: "/assistant", label: "Assistant" },
];

/* The reference design puts a notification bell and a signed-in user chip
   here. Both are omitted deliberately: this platform has no auth and no
   user accounts, so either one would be a painted-on control representing
   a person who does not exist. The search is wired to the real customer
   lookup instead of being decorative. */
export default function TopBar() {
  const router = useRouter();
  const pathname = usePathname();
  const [customerId, setCustomerId] = useState("");

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = customerId.trim().toUpperCase();
    if (!id) return;
    router.push(`/customers/${encodeURIComponent(id)}`);
  }

  return (
    <header
      className="sticky top-0 z-20 border-b"
      style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}
    >
      <div className="flex h-16 items-center gap-4 px-4 sm:px-6">
        <form onSubmit={onSubmit} className="relative flex-1 max-w-md" role="search">
          <label htmlFor="customer-search" className="sr-only">
            Look up a customer by ID
          </label>
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2"
            style={{ color: "var(--text-muted)" }}
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.2-3.2" />
          </svg>
          <input
            id="customer-search"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="Look up a customer ID, e.g. CUST0000000001"
            className="w-full rounded-lg border py-2 pl-9 pr-3 text-[13px] outline-none transition-colors focus:border-[color:var(--brand)]"
            style={{
              borderColor: "var(--border)",
              background: "var(--surface-sunken)",
              color: "var(--text-primary)",
            }}
          />
        </form>

        <div className="ml-auto hidden items-center gap-2 sm:flex">
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11.5px] font-medium"
            style={{ background: "var(--brand-tint)", color: "var(--brand-strong)" }}
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--brand)" }} aria-hidden="true" />
            1,000,000 customers
          </span>
        </div>
      </div>

      {/* The sidebar is hidden below lg, so the same routes ride along here. */}
      <nav className="flex gap-1 overflow-x-auto border-t px-4 py-2 lg:hidden" style={{ borderColor: "var(--border)" }}>
        {NAV.map((item) => {
          const active = pathname === item.href || pathname?.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className="whitespace-nowrap rounded-md px-2.5 py-1.5 text-[12.5px] font-medium"
              style={{
                color: active ? "var(--brand-strong)" : "var(--text-secondary)",
                background: active ? "var(--brand-tint)" : "transparent",
              }}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
