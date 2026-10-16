"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getOverviewStats, getSegmentRates, ApiError } from "@/lib/api-client";
import type { FeatureImportance, SegmentBucket, SegmentRatesResponse } from "@/lib/types";
import ChurnBarChart from "@/components/ChurnBarChart";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

// Only contract and tenure_bucket are filterable on the Customers search
// endpoint (see src/model/api.py's search_customers) - those two get
// click-to-drill-down. The demographic columns are real segment data but
// have no backing filter, so their bars stay informational rather than
// linking to a page that can't actually filter by them.
const ACCOUNT_COLUMNS = [
  { column: "contract", label: "Churn rate by contract", clickable: true },
  { column: "tenure_bucket", label: "Churn rate by tenure", clickable: true },
];
const DEMOGRAPHIC_COLUMNS = [
  { column: "education", label: "Churn rate by education", clickable: false },
  { column: "marital_status", label: "Churn rate by marital status", clickable: false },
  { column: "payment_method", label: "Churn rate by payment method", clickable: false },
  { column: "gender", label: "Churn rate by gender", clickable: false },
];
const SEGMENT_COLUMNS = [...ACCOUNT_COLUMNS, ...DEMOGRAPHIC_COLUMNS];

function statStrip(data: SegmentRatesResponse) {
  if (data.buckets.length === 0) return null;
  const totalCustomers = data.buckets.reduce((sum: number, b: SegmentBucket) => sum + b.n_customers, 0);
  const highest = data.buckets.reduce((a, b) => (b.churn_rate > a.churn_rate ? b : a));
  const lowest = data.buckets.reduce((a, b) => (b.churn_rate < a.churn_rate ? b : a));
  return { totalCustomers, highest, lowest };
}

export default function SegmentsPage() {
  const router = useRouter();
  const [segments, setSegments] = useState<Record<string, SegmentRatesResponse>>({});
  const [importances, setImportances] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getOverviewStats(), ...SEGMENT_COLUMNS.map((s) => getSegmentRates(s.column))])
      .then(([stats, ...segmentResults]) => {
        if (cancelled) return;
        setImportances(stats.top_feature_importances);
        const bySlug: Record<string, SegmentRatesResponse> = {};
        SEGMENT_COLUMNS.forEach((s, i) => {
          bySlug[s.column] = segmentResults[i];
        });
        setSegments(bySlug);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load segment data.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <ErrorState message={error} />;
  if (loading) return <LoadingState />;

  function renderSegmentCard(s: { column: string; label: string; clickable: boolean }) {
    const data = segments[s.column];
    const stats = data ? statStrip(data) : null;
    return (
      <Card key={s.column} title={s.label}>
        {stats && (
          <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-[11.5px]" style={{ color: "var(--text-secondary)" }}>
            <span>
              <span style={{ color: "var(--text-muted)" }}>Total: </span>
              {stats.totalCustomers.toLocaleString()} customers
            </span>
            <span>
              <span style={{ color: "var(--status-critical)" }}>Highest: </span>
              {stats.highest.key.replaceAll("_", " ")} ({(stats.highest.churn_rate * 100).toFixed(1)}%)
            </span>
            <span>
              <span style={{ color: "var(--status-good)" }}>Lowest: </span>
              {stats.lowest.key.replaceAll("_", " ")} ({(stats.lowest.churn_rate * 100).toFixed(1)}%)
            </span>
          </div>
        )}
        {data && (
          <ChurnBarChart
            data={data.buckets.map((b) => ({ label: b.key, value: b.churn_rate }))}
            valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            height={220}
            onBarClick={s.clickable ? (label) => router.push(`/customers?${s.column}=${encodeURIComponent(label)}`) : undefined}
          />
        )}
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Customer Intelligence"
        title="Segments"
        description="Churn rate broken out across account and demographic dimensions. Click a contract or tenure bar to see that segment's customers."
      />

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Account segments
        </h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">{ACCOUNT_COLUMNS.map(renderSegmentCard)}</div>
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Demographics
        </h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">{DEMOGRAPHIC_COLUMNS.map(renderSegmentCard)}</div>
      </div>

      <Card title="What the model weighs most">
        <ChurnBarChart
          data={importances.map((f) => ({ label: f.feature.replaceAll("_", " "), value: f.importance }))}
          valueFormatter={(v) => v.toFixed(0)}
          layout="horizontal"
          height={420}
        />
      </Card>
    </div>
  );
}
