"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getAtRisk, postOutreachDraft, ApiError } from "@/lib/api-client";
import type { AtRiskCustomer, AtRiskListResponse, OutreachDraftResponse } from "@/lib/types";
import ThresholdSlider from "@/components/ThresholdSlider";
import PaginationControls from "@/components/PaginationControls";
import ShapFactorList from "@/components/ShapFactorList";
import PageHeader from "@/components/PageHeader";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

const PAGE_SIZE = 25;

export default function AtRiskPage() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<AtRiskListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, OutreachDraftResponse | "loading" | "error">>({});

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
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load at-risk customers.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold, offset]);

  function generateDraft(customerId: string) {
    setDrafts((prev) => ({ ...prev, [customerId]: "loading" }));
    postOutreachDraft(customerId)
      .then((draft) => setDrafts((prev) => ({ ...prev, [customerId]: draft })))
      .catch(() => setDrafts((prev) => ({ ...prev, [customerId]: "error" })));
  }

  function exportCsv() {
    if (!data) return;
    const header = "customer_id,churn_probability,contract,tenure,monthlycharges,recommended_action\n";
    const rows = data.customers
      .map((c: AtRiskCustomer) => [c.customer_id, c.churn_probability, c.contract ?? "", c.tenure ?? "", c.monthlycharges ?? "", c.recommended_action ?? ""].join(","))
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "at_risk_customers.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  if (error) return <ErrorState message={error} />;

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Retention Command Center"
        title="At-risk customers"
        description="Ranked by churn probability, with the factors driving each one and a next-best action."
        action={
          <div className="flex flex-wrap items-center gap-3">
            <ThresholdSlider value={threshold ?? 0.1} onChange={(v) => { setThreshold(v); setOffset(0); }} />
            <button
              type="button"
              onClick={exportCsv}
              disabled={!data || data.customers.length === 0}
              className="rounded-lg px-3.5 py-2 text-[12.5px] font-semibold text-white transition-opacity disabled:opacity-40"
              style={{ background: "var(--brand)" }}
            >
              Export CSV
            </button>
          </div>
        }
      />

      {loading && !data ? (
        <LoadingState label="Scoring the full customer population - this can take a moment on first load..." />
      ) : data ? (
        <>
          <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left" style={{ color: "var(--text-muted)", background: "var(--surface-card)" }}>
                  <th className="px-3 py-2 font-medium">Customer</th>
                  <th className="px-3 py-2 font-medium">Churn prob.</th>
                  <th className="px-3 py-2 font-medium">Contract</th>
                  <th className="px-3 py-2 font-medium">Tenure</th>
                  <th className="px-3 py-2 font-medium">Monthly charges</th>
                  <th className="px-3 py-2 font-medium">Risk factors</th>
                  <th className="px-3 py-2 font-medium">Recommended action</th>
                  <th className="px-3 py-2 font-medium">AI outreach</th>
                </tr>
              </thead>
              <tbody>
                {data.customers.map((c) => {
                  const draft = drafts[c.customer_id];
                  return (
                    <tr key={c.customer_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="px-3 py-2">
                        <Link href={`/customers/${c.customer_id}`} className="font-medium underline" style={{ color: "var(--series-1)" }}>
                          {c.customer_id}
                        </Link>
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 overflow-hidden rounded-full" style={{ background: "var(--gridline)" }}>
                            <div className="h-full" style={{ width: `${c.churn_probability * 100}%`, background: "var(--status-critical)" }} />
                          </div>
                          <span className="tabular-nums">{(c.churn_probability * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>{c.contract ?? "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.tenure ?? "-"}</td>
                      <td className="px-3 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>{c.monthlycharges ? `$${c.monthlycharges.toFixed(2)}` : "-"}</td>
                      <td className="px-3 py-2"><ShapFactorList factors={c.key_risk_factors} /></td>
                      <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>{c.recommended_action?.replaceAll("_", " ") ?? "-"}</td>
                      <td className="px-3 py-2">
                        {!draft && (
                          <button
                            type="button"
                            onClick={() => generateDraft(c.customer_id)}
                            className="rounded-md border px-2 py-1 text-xs"
                            style={{ borderColor: "var(--border)", color: "var(--series-1)" }}
                          >
                            Generate
                          </button>
                        )}
                        {draft === "loading" && <span className="text-xs" style={{ color: "var(--text-muted)" }}>Generating...</span>}
                        {draft === "error" && <span className="text-xs" style={{ color: "var(--status-critical)" }}>Failed</span>}
                        {draft && typeof draft === "object" && (
                          <p className="max-w-xs text-xs" style={{ color: "var(--text-secondary)" }}>
                            {draft.draft ?? "No service to recommend."}
                          </p>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <PaginationControls offset={offset} limit={PAGE_SIZE} total={data.total_at_risk} onChange={setOffset} />
        </>
      ) : null}
    </div>
  );
}
