"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { getAtRisk, ApiError } from "@/lib/api-client";
import type { AtRiskCustomer, AtRiskListResponse } from "@/lib/types";
import { riskTier, type RiskTier } from "@/components/RiskBadge";
import RiskBadge from "@/components/RiskBadge";
import ThresholdSlider from "@/components/ThresholdSlider";
import PaginationControls from "@/components/PaginationControls";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const PAGE_SIZE = 25;
const STORAGE_KEY = "ci-recommendation-status";

type Status = "pending" | "in_progress" | "completed";
const STATUS_LABEL: Record<Status, string> = { pending: "Pending", in_progress: "In progress", completed: "Completed" };
const STATUS_ORDER: Status[] = ["pending", "in_progress", "completed"];

function loadStatuses(): Record<string, Status> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, Status>) : {};
  } catch {
    return {};
  }
}

const TIER_LABEL: Record<RiskTier, string> = { high: "High priority", medium: "Medium priority", low: "Low priority" };
const TIER_ORDER: RiskTier[] = ["high", "medium", "low"];

export default function RecommendationsPage() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<AtRiskListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Record<string, Status>>(loadStatuses);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getAtRisk(threshold ?? undefined, PAGE_SIZE, offset)
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setThreshold((prev) => prev ?? res.threshold);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load recommendations.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold, offset]);

  function setStatus(customerId: string, status: Status) {
    setStatuses((prev) => {
      const next = { ...prev, [customerId]: status };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // best-effort only - never block the UI on a storage write failing
      }
      return next;
    });
  }

  const grouped = useMemo(() => {
    const byTier: Record<RiskTier, AtRiskCustomer[]> = { high: [], medium: [], low: [] };
    for (const c of data?.customers ?? []) byTier[riskTier(c.churn_probability)].push(c);
    return byTier;
  }, [data]);

  if (error) return <ErrorState message={error} />;

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Customer Intelligence"
        title="Recommendations"
        description="At-risk customers grouped by priority, with the recommended retention action for each. Sorted by churn probability across the whole at-risk list."
        action={<ThresholdSlider value={threshold ?? 0.1} onChange={(v) => { setThreshold(v); setOffset(0); }} />}
      />

      <p
        className="rounded-lg px-3.5 py-2.5 text-[12px]"
        style={{ background: "var(--surface-sunken)", color: "var(--text-muted)" }}
      >
        Status is tracked in this browser only (not yet saved to the backend) - it will reset if you clear site data
        or switch devices.
      </p>

      {loading && !data ? (
        <LoadingState label="Scoring the full customer population - this can take a moment on first load..." />
      ) : data ? (
        <>
          {data.customers.length === 0 ? (
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>No customers above the current threshold.</p>
          ) : (
            <div className="space-y-5">
              {TIER_ORDER.filter((tier) => grouped[tier].length > 0).map((tier) => (
                <div key={tier}>
                  <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    {TIER_LABEL[tier]} <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>({grouped[tier].length} on this page)</span>
                  </h2>
                  <Card bodyClassName="px-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                            <th className="px-3 py-2 font-medium">Customer</th>
                            <th className="px-3 py-2 font-medium">Risk</th>
                            <th className="px-3 py-2 font-medium">Reason</th>
                            <th className="px-3 py-2 font-medium">Recommended action</th>
                            <th className="px-3 py-2 font-medium">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {grouped[tier].map((c) => {
                            const status = statuses[c.customer_id] ?? "pending";
                            const topFactor = c.key_risk_factors[0];
                            return (
                              <tr key={c.customer_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                                <td className="px-3 py-2">
                                  <Link href={`/customers/${c.customer_id}`} className="font-medium underline" style={{ color: "var(--series-1)" }}>
                                    {c.customer_id}
                                  </Link>
                                </td>
                                <td className="px-3 py-2">
                                  <RiskBadge churnProbability={c.churn_probability} />
                                  <span className="ml-2 tabular-nums text-xs" style={{ color: "var(--text-muted)" }}>
                                    {(c.churn_probability * 100).toFixed(1)}%
                                  </span>
                                </td>
                                <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>
                                  {topFactor ? topFactor.feature.replaceAll("_", " ") : "-"}
                                </td>
                                <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>
                                  {c.recommended_action?.replaceAll("_", " ") ?? "-"}
                                </td>
                                <td className="px-3 py-2">
                                  <select
                                    value={status}
                                    onChange={(e) => setStatus(c.customer_id, e.target.value as Status)}
                                    className="rounded-md border px-2 py-1 text-xs"
                                    style={{ borderColor: "var(--border)", background: "var(--surface-card)", color: "var(--text-primary)" }}
                                  >
                                    {STATUS_ORDER.map((s) => (
                                      <option key={s} value={s}>{STATUS_LABEL[s]}</option>
                                    ))}
                                  </select>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          )}
          <PaginationControls offset={offset} limit={PAGE_SIZE} total={data.total_at_risk} onChange={setOffset} />
        </>
      ) : null}
    </div>
  );
}
