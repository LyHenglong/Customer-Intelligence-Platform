"use client";

import { useEffect, useState } from "react";
import { getOverviewStats, getRevenueAtRiskBySegment, getSegmentRates, ApiError } from "@/lib/api-client";
import type { OverviewStatsResponse, RevenueAtRiskResponse, SegmentRatesResponse } from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import Card from "@/components/Card";
import ChurnBarChart from "@/components/ChurnBarChart";
import ThresholdSlider from "@/components/ThresholdSlider";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const kpiIcon = (path: React.ReactNode) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" className="h-[18px] w-[18px]">
    {path}
  </svg>
);

const ICONS = {
  customers: kpiIcon(<><circle cx="9" cy="8" r="3.2" /><path d="M2.5 20a6.5 6.5 0 0 1 13 0" /><path d="M16.5 5.6a3.2 3.2 0 0 1 0 6.3" /><path d="M18 14.3a6.5 6.5 0 0 1 3.5 5.7" /></>),
  churn: kpiIcon(<><path d="M3 17l5.5-5.5 3.5 3.5L21 6" /><path d="M15 6h6v6" /></>),
  atRisk: kpiIcon(<><path d="M12 3.5 2.8 19.5h18.4L12 3.5Z" /><path d="M12 10v4" /><path d="M12 17.2h.01" /></>),
  revenue: kpiIcon(<><circle cx="12" cy="12" r="8.5" /><path d="M12 7.2v9.6" /><path d="M14.3 9.6a2.6 2.6 0 0 0-2.3-1.2c-1.4 0-2.4.8-2.4 1.9 0 2.6 4.8 1.4 4.8 4 0 1.2-1.1 2-2.5 2a2.8 2.8 0 0 1-2.5-1.3" /></>),
  model: kpiIcon(<><path d="M3 20h18" /><path d="M6 20v-6" /><path d="M11 20V7" /><path d="M16 20v-9" /><path d="M21 20V4" /></>),
};

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
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[13px]" style={{ color: "var(--text-secondary)" }}>
            Retention Command Center
          </p>
          <h1 className="mt-0.5 text-[26px] font-bold leading-tight tracking-tight" style={{ color: "var(--text-primary)" }}>
            Here&rsquo;s your retention overview
          </h1>
          <p className="mt-1 text-[13px]" style={{ color: "var(--text-secondary)" }}>
            Spot at-risk customers, see where churn concentrates, and act before they leave.
          </p>
        </div>
        <ThresholdSlider value={threshold ?? stats.threshold_used} onChange={setThreshold} />
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {/* Labels kept to one line each - a wrapping label was pushing its
            card taller than the rest of the row. */}
        <KpiCard label="Total customers" value={stats.total_customers.toLocaleString()} icon={ICONS.customers} tone="info" sublabel="in the warehouse" />
        <KpiCard
          label="Churn rate"
          value={`${(stats.historical_churn_rate * 100).toFixed(1)}%`}
          icon={ICONS.churn}
          tone="warning"
          sublabel="historical, observed"
        />
        <KpiCard
          label="At-risk now"
          value={stats.at_risk_count.toLocaleString()}
          icon={ICONS.atRisk}
          tone="danger"
          sublabel={`${((stats.at_risk_count / stats.total_customers) * 100).toFixed(1)}% of all customers`}
        />
        <KpiCard
          label="Revenue at risk"
          value={compactCurrency(stats.revenue_at_risk)}
          icon={ICONS.revenue}
          tone="brand"
          sublabel="per month, recurring"
        />
        <KpiCard
          label="Model AUC"
          value={stats.model_auc ? stats.model_auc.toFixed(3) : "-"}
          icon={ICONS.model}
          tone="violet"
          sublabel={stats.model_version ?? undefined}
        />
      </div>

      <div>
        <h2 className="mb-3 text-[13.5px] font-semibold" style={{ color: "var(--text-primary)" }}>
          Where churn concentrates
        </h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {SEGMENT_COLUMNS.map((s) => {
            const data = segments[s.column];
            return (
              <Card key={s.column} title={s.label}>
                {data ? (
                  <ChurnBarChart
                    data={data.buckets.map((b) => ({ label: b.key, value: b.churn_rate }))}
                    valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    height={200}
                  />
                ) : (
                  <LoadingState />
                )}
              </Card>
            );
          })}
        </div>
      </div>

      <Card
        title="Revenue at risk by segment"
        action={
          <span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            Monthly recurring revenue, whole at-risk population
          </span>
        }
      >
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
      </Card>
    </div>
  );
}
