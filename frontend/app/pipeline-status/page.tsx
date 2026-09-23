"use client";

import { useEffect, useState } from "react";
import { getPipelineStatus, ApiError } from "@/lib/api-client";
import type { PipelineStatusResponse } from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import PageHeader from "@/components/PageHeader";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const SEVERITY_COLOR: Record<string, string> = {
  significant: "var(--status-critical)",
  moderate: "var(--status-warning)",
  stable: "var(--status-good)",
};

export default function PipelineStatusPage() {
  const [data, setData] = useState<PipelineStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getPipelineStatus()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load pipeline status.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <ErrorState message={error} />;
  if (loading || !data) return <LoadingState />;

  const drifted = data.drift.filter((d) => d.severity === "significant");
  const watching = data.drift.filter((d) => d.severity === "moderate");
  const maxPsi = data.drift.length > 0 ? Math.max(...data.drift.map((d) => d.psi)) : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Retention Command Center"
        title="Pipeline status"
        description="Batch ingestion, feature drift, and where the next conditional retrain sits."
      />

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Operations
        </h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <KpiCard label="Batches ingested" value={`${data.batches_ingested} / ${data.total_simulated_batches}`} />
          <KpiCard label="Batches to next retrain" value={String(data.batches_to_next_retrain)} sublabel={`every ${data.retrain_every_n_batches} batches`} />
          <KpiCard label="Model versions trained" value={String(data.model_versions_trained)} />
        </div>
      </div>

      <div className="rounded-lg border" style={{ borderColor: "var(--border)" }}>
        <div className="border-b px-4 py-2 text-xs font-medium" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
          Ingestion log
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                <th className="px-3 py-2 font-medium">Batch</th>
                <th className="px-3 py-2 font-medium">Rows loaded</th>
                <th className="px-3 py-2 font-medium">Loaded at</th>
                <th className="px-3 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.ingestion_log.map((row) => (
                <tr key={row.batch_file} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="px-3 py-2">{row.batch_file}</td>
                  <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{row.rows_loaded.toLocaleString()}</td>
                  <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>{row.loaded_at}</td>
                  <td className="px-3 py-2" style={{ color: row.status === "success" ? "var(--status-good)" : "var(--status-critical)" }}>{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Feature drift (PSI)
        </h2>
        <div className="mb-3 grid grid-cols-3 gap-4">
          <KpiCard label="Highest PSI" value={maxPsi.toFixed(3)} />
          <KpiCard label="Features drifted" value={String(drifted.length)} />
          <KpiCard label="Features to watch" value={String(watching.length)} />
        </div>
        {data.drift.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>No drift data yet.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                  <th className="px-3 py-2 font-medium">Feature</th>
                  <th className="px-3 py-2 font-medium">PSI</th>
                  <th className="px-3 py-2 font-medium">Severity</th>
                </tr>
              </thead>
              <tbody>
                {data.drift.map((row) => (
                  <tr key={row.feature} className="border-t" style={{ borderColor: "var(--border)" }}>
                    <td className="px-3 py-2">{row.feature}</td>
                    <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{row.psi.toFixed(4)}</td>
                    <td className="px-3 py-2 font-medium" style={{ color: SEVERITY_COLOR[row.severity] ?? "var(--text-secondary)" }}>{row.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {data.latest_retrain_summary && (
        <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
          <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Latest retrain summary
          </h2>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{data.latest_retrain_summary.summary_text}</p>
        </div>
      )}
    </div>
  );
}
