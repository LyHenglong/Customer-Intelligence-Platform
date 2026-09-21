"use client";

import { useEffect, useState } from "react";
import { getOverviewStats, getRevenueAtRiskBySegment, getSegmentRates, ApiError } from "@/lib/api-client";
import type { OverviewStatsResponse, RevenueAtRiskResponse, SegmentRatesResponse } from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import ChurnBarChart from "@/components/ChurnBarChart";
import ThresholdSlider from "@/components/ThresholdSlider";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const SEGMENT_COLUMNS = [
  { column: "contract", label: "Churn rate by contract type" },
  { column: "tenure_bucket", label: "Churn rate by tenure" },
  { column: "total_active_services", label: "Churn rate by service bundle size" },
];

function compactCurrency(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

export default function OverviewPage() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [stats, setStats] = useState<OverviewStatsResponse | null>(null);
  const [segments, setSegments] = useState<Record<string, SegmentRatesResponse>>({});
  const [revenueAtRisk, setRevenueAtRisk] = useState<RevenueAtRiskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      getOverviewStats(threshold ?? undefined),
      getRevenueAtRiskBySegment(threshold ?? undefined),
      ...SEGMENT_COLUMNS.map((s) => getSegmentRates(s.column)),
    ])
      .then(([statsRes, revenueRes, ...segmentResults]) => {
        if (cancelled) return;
        setStats(statsRes);
        setRevenueAtRisk(revenueRes);
        setThreshold((prev) => prev ?? statsRes.threshold_used);
        const bySlug: Record<string, SegmentRatesResponse> = {};
        SEGMENT_COLUMNS.forEach((s, i) => {
          bySlug[s.column] = segmentResults[i];
        });
        setSegments(bySlug);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load overview data.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold]);

  if (error) return <ErrorState message={error} />;
  if (loading && !stats) return <LoadingState label="Scoring the full customer population - this can take a moment on first load..." />;
  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
          Overview
        </h1>
        <ThresholdSlider value={threshold ?? stats.threshold_used} onChange={setThreshold} />
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <KpiCard label="Total customers" value={stats.total_customers.toLocaleString()} />
        <KpiCard label="Historical churn rate" value={`${(stats.historical_churn_rate * 100).toFixed(1)}%`} />
        <KpiCard label="At-risk now" value={stats.at_risk_count.toLocaleString()} />
        <KpiCard label="Revenue at risk / mo" value={compactCurrency(stats.revenue_at_risk)} />
        <KpiCard label="Model AUC" value={stats.model_auc ? stats.model_auc.toFixed(3) : "-"} sublabel={stats.model_version ?? undefined} />
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Where churn concentrates
        </h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {SEGMENT_COLUMNS.map((s) => {
            const data = segments[s.column];
            return (
              <div key={s.column} className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
                <h3 className="mb-2 text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                  {s.label}
                </h3>
                {data ? (
                  <ChurnBarChart
                    data={data.buckets.map((b) => ({ label: b.key, value: b.churn_rate }))}
                    valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    height={200}
                  />
                ) : (
                  <LoadingState />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Revenue at risk by segment
        </h2>
        {revenueAtRisk && revenueAtRisk.buckets.length > 0 ? (
          <ChurnBarChart
            data={revenueAtRisk.buckets.map((b) => ({ label: b.segment, value: b.revenue_at_risk }))}
            valueFormatter={compactCurrency}
            layout="horizontal"
            height={220}
          />
        ) : (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            No at-risk customers above the current threshold.
          </p>
        )}
      </div>
    </div>
  );
}
