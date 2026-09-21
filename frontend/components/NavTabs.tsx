"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/overview", label: "Overview" },
  { href: "/at-risk", label: "At-Risk Customers" },
  { href: "/customers", label: "Customers" },
  { href: "/segments", label: "Segments" },
  { href: "/model-performance", label: "Model Performance" },
  { href: "/pipeline-status", label: "Pipeline Status" },
  { href: "/assistant", label: "AI Assistant" },
];

export default function NavTabs() {
  const pathname = usePathname();

  return (
    <nav className="flex gap-1 overflow-x-auto">
      {TABS.map((tab) => {
        const active = pathname === tab.href || (tab.href !== "/overview" && pathname?.startsWith(tab.href));
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className="whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors"
            style={{
              color: active ? "var(--text-primary)" : "var(--text-secondary)",
              background: active ? "var(--surface-page)" : "transparent",
            }}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
