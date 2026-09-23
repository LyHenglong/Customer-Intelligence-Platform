"use client";

import { useEffect, useState } from "react";
import { Line, LineChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getModelHistory, ApiError } from "@/lib/api-client";
import type { ModelVersionMetadata } from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

/** "20260909T073056Z" -> "09-09 07:30", so same-day retrains stay distinguishable. */
function formatVersionTick(version: string) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(version);
  return m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : version;
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const labels = ["Retained", "Churned"];
  const max = Math.max(...matrix.flat());
  return (
    <table className="text-sm">
      <thead>
        <tr>
          <th />
          <th className="px-3 pb-1 font-medium" style={{ color: "var(--text-muted)" }} colSpan={2}>
            Predicted
          </th>
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            {i === 0 && (
              <th rowSpan={2} className="pr-2 text-left align-middle font-medium" style={{ color: "var(--text-muted)" }}>
                <span className="[writing-mode:vertical-lr] rotate-180">Actual</span>
              </th>
            )}
            {row.map((v, j) => (
              <td key={j} className="border p-4 text-center tabular-nums" style={{ borderColor: "var(--border)", background: `color-mix(in srgb, var(--series-1) ${(v / max) * 40}%, var(--surface-card))` }}>
                <div className="font-semibold" style={{ color: "var(--text-primary)" }}>{v.toLocaleString()}</div>
                <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>{labels[i]} → {labels[j]}</div>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelPerformancePage() {
  const [history, setHistory] = useState<ModelVersionMetadata[]>([]);
  const [servingVersion, setServingVersion] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelHistory()
      .then((res) => {
        if (!cancelled) {
          setHistory(res.versions);
          setServingVersion(res.serving_version);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load model history.");
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
  if (history.length === 0) return <ErrorState message="No trained model versions found." />;

  // The version actually being served, which is not necessarily the newest
  // on disk - the registry can resolve an older artifact (e.g. an MLflow
  // "champion" alias left pinned to a previous version after a retrain
  // saved but failed to register). Labelling the newest file "current
  // production model" misreported the live threshold and confusion matrix.
  const serving = history.find((h) => h.version === servingVersion);
  const latest = serving ?? history[history.length - 1];
  const isStale = Boolean(servingVersion) && history[history.length - 1]?.version !== servingVersion;
  // Full version as the category key, not a date-only slice: several
  // retrains can land on the same day, and duplicate category keys make
  // Recharts drop line segments between them (and render a row of
  // identical axis ticks). formatVersionTick handles readability instead.
  const trend = history.map((h) => ({
    version: h.version,
    roc_auc: h.roc_auc ?? null,
    f1_churn: h.f1_churn ?? null,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Model Performance
      </h1>

      <div>
        <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Current production model - {latest.version}
        </h2>
        {isStale && (
          <p className="mb-2 text-xs" style={{ color: "var(--status-warning)" }}>
            A newer artifact ({history[history.length - 1].version}) exists but is not
            being served - the model registry resolves to the version above. Check the
            MLflow &quot;champion&quot; alias if that is unintended.
          </p>
        )}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <KpiCard label="Precision (churn)" value={latest.precision_churn?.toFixed(3) ?? "-"} />
          <KpiCard label="Recall (churn)" value={latest.recall_churn?.toFixed(3) ?? "-"} />
          <KpiCard label="F1 (churn)" value={latest.f1_churn?.toFixed(3) ?? "-"} />
          <KpiCard label="Decision threshold" value={latest.threshold?.toFixed(3) ?? "-"} />
        </div>
        {latest.threshold_rationale && (
          <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
            {latest.threshold_rationale}
          </p>
        )}
      </div>

      {latest.confusion_matrix && (
        <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
          <h2 className="mb-3 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Confusion matrix (at threshold)
          </h2>
          <ConfusionMatrix matrix={latest.confusion_matrix} />
        </div>
      )}

      {history.length > 1 && (
        <div className="rounded-lg border p-4" style={{ borderColor: "var(--border)", background: "var(--surface-card)" }}>
          <h2 className="mb-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Performance across retrains
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trend} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--gridline)" />
              <XAxis dataKey="version" tickFormatter={formatVersionTick} tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: "var(--surface-card)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
              <Line type="monotone" dataKey="roc_auc" name="ROC AUC" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="f1_churn" name="F1 (churn)" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
