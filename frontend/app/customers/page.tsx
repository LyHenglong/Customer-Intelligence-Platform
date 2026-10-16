"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { searchCustomers, ApiError } from "@/lib/api-client";
import type { CustomerSearchFilters, CustomerSearchResult } from "@/lib/types";
import PaginationControls from "@/components/PaginationControls";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const PAGE_SIZE = 25;

// useSearchParams() (to seed filters from a Segments-page click-through)
// requires a Suspense boundary or the App Router bails the whole page out
// of static rendering with a build error - this page has no useful
// pre-search-params content anyway, so the fallback is just the loader.
export default function CustomersPage() {
  return (
    <Suspense fallback={<LoadingState />}>
      <CustomersPageInner />
    </Suspense>
  );
}

function CustomersPageInner() {
  // Seeded from ?contract=X or ?tenure_bucket=X, e.g. a click-through from
  // the Segments page's account-segment bar charts.
  const searchParams = useSearchParams();
  const [filters, setFilters] = useState<CustomerSearchFilters>(() => ({
    contract: searchParams.get("contract") ?? undefined,
    tenure_bucket: searchParams.get("tenure_bucket") ?? undefined,
  }));
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<CustomerSearchResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    searchCustomers({ ...filters, limit: PAGE_SIZE, offset })
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Search failed.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [filters, offset]);

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Customer Intelligence"
        title="Customers"
        description="Search and filter the full customer base, then open anyone for a complete risk profile."
      />

      <div className="flex flex-wrap gap-3">
        <select
          value={filters.contract ?? ""}
          onChange={(e) => {
            setFilters((f) => ({ ...f, contract: e.target.value || undefined }));
            setOffset(0);
          }}
          className="rounded-md border px-3 py-1.5 text-sm"
          style={{ borderColor: "var(--border)", background: "var(--surface-card)", color: "var(--text-primary)" }}
        >
          <option value="">Any contract</option>
          <option value="month_to_month">Month-to-month</option>
          <option value="one_year">One year</option>
          <option value="two_year">Two year</option>
        </select>
        <select
          value={filters.tenure_bucket ?? ""}
          onChange={(e) => {
            setFilters((f) => ({ ...f, tenure_bucket: e.target.value || undefined }));
            setOffset(0);
          }}
          className="rounded-md border px-3 py-1.5 text-sm"
          style={{ borderColor: "var(--border)", background: "var(--surface-card)", color: "var(--text-primary)" }}
        >
          <option value="">Any tenure</option>
          <option value="new_0_6mo">New (0-6mo)</option>
          <option value="established_6_24mo">Established (6-24mo)</option>
          <option value="loyal_24mo_plus">Loyal (24mo+)</option>
        </select>
        <input
          type="number"
          placeholder="Min churn probability"
          step={0.05}
          min={0}
          max={1}
          className="w-48 rounded-md border px-3 py-1.5 text-sm"
          style={{ borderColor: "var(--border)", background: "var(--surface-card)", color: "var(--text-primary)" }}
          onChange={(e) => {
            const v = e.target.value ? Number(e.target.value) : undefined;
            setFilters((f) => ({ ...f, min_churn_probability: v }));
            setOffset(0);
          }}
        />
      </div>

      {error && <ErrorState message={error} />}
      {loading && !data ? (
        <LoadingState />
      ) : data ? (
        <>
          <Card bodyClassName="px-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                    <th className="px-3 py-2 font-medium">Customer</th>
                    <th className="px-3 py-2 font-medium">Contract</th>
                    <th className="px-3 py-2 font-medium">Tenure</th>
                    <th className="px-3 py-2 font-medium">Monthly charges</th>
                    <th className="px-3 py-2 font-medium">Satisfaction</th>
                    <th className="px-3 py-2 font-medium">Complaints</th>
                  </tr>
                </thead>
                <tbody>
                  {data.customers.map((c) => (
                    <tr key={c.customer_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="px-3 py-2">
                        <Link href={`/customers/${c.customer_id}`} className="font-medium underline" style={{ color: "var(--series-1)" }}>
                          {c.customer_id}
                        </Link>
                      </td>
                      <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>{c.contract ?? "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.tenure ?? "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.monthlycharges ? `$${c.monthlycharges.toFixed(2)}` : "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.customer_satisfaction ?? "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.num_complaints ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          {data.truncated && (
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              More customers matched than could be scored - refine your filters for an exact count.
            </p>
          )}
          <PaginationControls offset={offset} limit={PAGE_SIZE} total={data.total_matched} onChange={setOffset} />
        </>
      ) : null}
    </div>
  );
}
