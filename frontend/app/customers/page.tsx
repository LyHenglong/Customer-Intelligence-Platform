"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { searchCustomers, ApiError } from "@/lib/api-client";
import type { CustomerSearchFilters, CustomerSearchResult } from "@/lib/types";
import PaginationControls from "@/components/PaginationControls";
import PageHeader from "@/components/PageHeader";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const PAGE_SIZE = 25;

export default function CustomersPage() {
  const [filters, setFilters] = useState<CustomerSearchFilters>({});
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
        eyebrow="Retention Command Center"
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
          <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left" style={{ color: "var(--text-muted)", background: "var(--surface-card)" }}>
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
