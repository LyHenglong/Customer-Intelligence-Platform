"use client";

import { useEffect, useState } from "react";
import { getOverviewStats, getSegmentRates, ApiError } from "@/lib/api-client";
import type { FeatureImportance, SegmentRatesResponse } from "@/lib/types";
import ChurnBarChart from "@/components/ChurnBarChart";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const SEGMENT_COLUMNS = [
  { column: "education", label: "Churn rate by education" },
  { column: "marital_status", label: "Churn rate by marital status" },
  { column: "payment_method", label: "Churn rate by payment method" },
  { column: "gender", label: "Churn rate by gender" },
];

export default function SegmentsPage() {
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

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Segments
      </h1>

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Demographics
        </h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {SEGMENT_COLUMNS.map((s) => {
            const data = segments[s.column];
            return (
              <div key={s.column} className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
                <h3 className="mb-2 text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                  {s.label}
                </h3>
                {data && (
                  <ChurnBarChart
                    data={data.buckets.map((b) => ({ label: b.key, value: b.churn_rate }))}
                    valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    height={220}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          What the model weighs most
        </h2>
        <ChurnBarChart
          data={importances.map((f) => ({ label: f.feature.replaceAll("_", " "), value: f.importance }))}
          valueFormatter={(v) => v.toFixed(0)}
          layout="horizontal"
          height={420}
        />
      </div>
    </div>
  );
}
