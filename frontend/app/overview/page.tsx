"use client";

import { useEffect, useState } from "react";
import {
  getOverviewStats,
  getRevenueAtRiskBySegment,
  getSegmentRates,
  getPipelineStatus,
  getModelHistory,
  ApiError,
} from "@/lib/api-client";
import type {
  OverviewStatsResponse,
  RevenueAtRiskResponse,
  SegmentRatesResponse,
  PipelineStatusResponse,
  ModelVersionMetadata,
} from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import Card from "@/components/Card";
import ChurnBarChart from "@/components/ChurnBarChart";
import DonutChart from "@/components/DonutChart";
import TrendAreaChart from "@/components/TrendAreaChart";
import FunnelChart from "@/components/FunnelChart";
import FeedList, { type FeedItem } from "@/components/FeedList";
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

function compactCurrency(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

const compactCount = (value: number) => (value >= 1000 ? `${(value / 1000).toFixed(0)}K` : String(value));

export default function OverviewPage() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [stats, setStats] = useState<OverviewStatsResponse | null>(null);
  const [contractRates, setContractRates] = useState<SegmentRatesResponse | null>(null);
  const [tenureRates, setTenureRates] = useState<SegmentRatesResponse | null>(null);
  const [revenueAtRisk, setRevenueAtRisk] = useState<RevenueAtRiskResponse | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatusResponse | null>(null);
  const [models, setModels] = useState<ModelVersionMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      getOverviewStats(threshold ?? undefined),
      getRevenueAtRiskBySegment(threshold ?? undefined),
      getSegmentRates("contract"),
      getSegmentRates("tenure_bucket"),
      getPipelineStatus(),
      getModelHistory(),
    ])
      .then(([statsRes, revenueRes, contractRes, tenureRes, pipelineRes, modelsRes]) => {
        if (cancelled) return;
        setStats(statsRes);
        setRevenueAtRisk(revenueRes);
        setContractRates(contractRes);
        setTenureRates(tenureRes);
        setPipeline(pipelineRes);
        setModels(modelsRes.versions);
        setThreshold((prev) => prev ?? statsRes.threshold_used);
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

  // Cumulative customers in the warehouse after each batch landed - the
  // real analogue of the reference's acquisition curve.
  const growth = (pipeline?.ingestion_log ?? []).reduce<{ label: string; value: number }[]>((acc, row, i) => {
    const previous = i > 0 ? acc[i - 1].value : 0;
    acc.push({ label: row.batch_file.replace(/^batch_|\.csv$/g, ""), value: previous + row.rows_loaded });
    return acc;
  }, []);

  // A survival funnel: how many customers have *reached at least* each
  // tenure stage, so it decreases monotonically the way a funnel must.
  //
  // The obvious version - one band per tenure bucket - is not a funnel at
  // all. Those are three co-existing populations, not sequential stages,
  // and plotting them this way produced bands reading 100% -> 186% -> 167%.
  // A funnel whose stages grow is telling the reader something false about
  // the data's shape, so the stages are cumulative instead.
  const bucketCount = (key: string) => tenureRates?.buckets.find((b) => b.key === key)?.n_customers ?? 0;
  const reached6 = bucketCount("established_6_24mo") + bucketCount("loyal_24mo_plus");
  const reached24 = bucketCount("loyal_24mo_plus");
  const funnel = tenureRates
    ? [
        { label: "Joined", value: tenureRates.buckets.reduce((sum, b) => sum + b.n_customers, 0) },
        { label: "Reached 6 months", value: reached6 },
        { label: "Reached 24 months", value: reached24 },
      ].filter((s) => s.value > 0)
    : [];

  const batchFeed: FeedItem[] = [...(pipeline?.ingestion_log ?? [])]
    .reverse()
    .slice(0, 5)
    .map((row) => ({
      id: row.batch_file,
      badge: row.batch_file.replace(/^batch_|\.csv$/g, ""),
      title: `${row.rows_loaded.toLocaleString()} rows loaded`,
      subtitle: row.batch_file,
      meta: row.loaded_at.slice(0, 10),
      status: row.status === "success" ? "good" : "critical",
    }));

  const modelFeed: FeedItem[] = [...models]
    .reverse()
    .slice(0, 5)
    .map((m) => ({
      id: m.version,
      badge: "v",
      title: m.roc_auc ? `AUC ${m.roc_auc.toFixed(3)}` : "metrics unavailable",
      subtitle: m.version,
      meta: m.f1_churn ? `F1 ${m.f1_churn.toFixed(3)}` : undefined,
      status: "neutral",
    }));

  const driftFeed: FeedItem[] = (pipeline?.drift ?? []).slice(0, 5).map((d) => ({
    id: d.feature,
    badge: "PSI",
    title: d.feature.replace(/_/g, " "),
    subtitle: `PSI ${d.psi.toFixed(4)}`,
    meta: d.severity,
    status: d.severity === "significant" ? "critical" : d.severity === "moderate" ? "warning" : "good",
  }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[13px]" style={{ color: "var(--text-secondary)" }}>
            Good day,
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

      {/* Row 1 - KPI strip */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <KpiCard label="Total customers" value={stats.total_customers.toLocaleString()} icon={ICONS.customers} tone="info" sublabel="in the warehouse" />
        <KpiCard label="Churn rate" value={`${(stats.historical_churn_rate * 100).toFixed(1)}%`} icon={ICONS.churn} tone="warning" sublabel="historical, observed" />
        <KpiCard
          label="At-risk now"
          value={stats.at_risk_count.toLocaleString()}
          icon={ICONS.atRisk}
          tone="danger"
          sublabel={`${((stats.at_risk_count / stats.total_customers) * 100).toFixed(1)}% of all customers`}
        />
        <KpiCard label="Revenue at risk" value={compactCurrency(stats.revenue_at_risk)} icon={ICONS.revenue} tone="brand" sublabel="per month, recurring" />
        <KpiCard label="Model AUC" value={stats.model_auc ? stats.model_auc.toFixed(3) : "-"} icon={ICONS.model} tone="violet" sublabel={stats.model_version ?? undefined} />
      </div>

      {/* Row 2 - ranked magnitudes beside a composition donut */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card
          title="Churn rate by contract"
          action={<span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>Whole customer base</span>}
        >
          {contractRates ? (
            <ChurnBarChart
              data={contractRates.buckets.map((b) => ({ label: b.key, value: b.churn_rate }))}
              valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              layout="horizontal"
              height={240}
            />
          ) : (
            <LoadingState />
          )}
        </Card>

        <Card
          title="Revenue at risk by contract"
          action={<span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>Share of monthly total</span>}
        >
          {revenueAtRisk && revenueAtRisk.buckets.length > 0 ? (
            <DonutChart
              data={revenueAtRisk.buckets.map((b) => ({ label: b.segment, value: b.revenue_at_risk }))}
              centerValue={compactCurrency(stats.revenue_at_risk)}
              centerLabel="at risk / mo"
              valueFormatter={compactCurrency}
            />
          ) : (
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              No at-risk customers above the current threshold.
            </p>
          )}
        </Card>
      </div>

      {/* Row 3 - change over time beside the retention funnel */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card
          title="Customer base growth"
          action={<span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>Cumulative, by batch</span>}
        >
          {growth.length > 0 ? (
            <TrendAreaChart data={growth} valueFormatter={compactCount} tooltipLabel="Customers" height={240} />
          ) : (
            <LoadingState />
          )}
        </Card>

        <Card
          title="Retention funnel by tenure"
          action={<span className="text-[11.5px]" style={{ color: "var(--text-muted)" }}>Share still active at each stage</span>}
        >
          {funnel.length > 0 ? (
            <div className="py-3">
              <FunnelChart stages={funnel} />
            </div>
          ) : (
            <LoadingState />
          )}
        </Card>
      </div>

      {/* Row 4 - three operational feeds */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Recent batches">
          <FeedList items={batchFeed} emptyMessage="No batches ingested yet." />
        </Card>
        <Card title="Model versions">
          <FeedList items={modelFeed} emptyMessage="No models trained yet." />
        </Card>
        <Card title="Feature drift">
          <FeedList items={driftFeed} emptyMessage="No drift measured yet." />
        </Card>
      </div>
    </div>
  );
}
