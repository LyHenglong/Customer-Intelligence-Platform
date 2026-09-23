"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getCustomer, postOutreachDraft, ApiError } from "@/lib/api-client";
import type { CustomerLookupResult, OutreachDraftResponse } from "@/lib/types";
import KpiCard from "@/components/KpiCard";
import Card from "@/components/Card";
import ShapFactorList from "@/components/ShapFactorList";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

export default function CustomerDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [customer, setCustomer] = useState<CustomerLookupResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<OutreachDraftResponse | "loading" | "error" | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCustomer(id)
      .then((res) => {
        if (!cancelled) setCustomer(res);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load customer.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  function generateDraft() {
    setDraft("loading");
    postOutreachDraft(id)
      .then(setDraft)
      .catch(() => setDraft("error"));
  }

  if (error) return <ErrorState message={error} />;
  if (loading) return <LoadingState />;
  if (!customer || !customer.found) {
    return <ErrorState message={`Customer ${id} not found.`} />;
  }

  const p = customer.profile;

  return (
    <div className="space-y-6">
      <div>
        <Link href="/customers" className="text-[12.5px] font-medium" style={{ color: "var(--brand-strong)" }}>
          ← Back to customers
        </Link>
        <h1 className="mt-1 text-[26px] font-bold leading-tight tracking-tight" style={{ color: "var(--text-primary)" }}>
          {customer.customer_id}
        </h1>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <KpiCard
          label="Churn probability"
          value={customer.churn_probability !== null ? `${(customer.churn_probability * 100).toFixed(1)}%` : "-"}
          sublabel={customer.risk_status ? `${customer.risk_status} risk (threshold ${customer.churn_threshold?.toFixed(2)})` : undefined}
          tone="danger"
        />
        <KpiCard label="Contract" value={p?.contract ?? "-"} tone="info" />
        <KpiCard label="Tenure" value={p?.tenure !== null && p?.tenure !== undefined ? `${p.tenure} mo` : "-"} tone="violet" />
        <KpiCard label="Monthly charges" value={p?.monthlycharges ? `$${p.monthlycharges.toFixed(2)}` : "-"} tone="brand" />
      </div>

      {customer.shap_factors.length > 0 && (
        <Card title="Risk factors">
          <ShapFactorList factors={customer.shap_factors} />
        </Card>
      )}

      {customer.recommendation.length > 0 && (
        <Card title="Recommended services">
          <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            {customer.recommendation.map((r) => (
              <li key={r.service}>
                {r.service.replaceAll("_", " ")} <span style={{ color: "var(--text-muted)" }}>(score {r.score.toFixed(2)})</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card
        title="AI explanation & outreach draft"
        action={
          !draft ? (
            <button
              type="button"
              onClick={generateDraft}
              className="rounded-lg px-3.5 py-2 text-[12px] font-semibold text-white"
              style={{ background: "var(--brand)" }}
            >
              Generate
            </button>
          ) : undefined
        }
      >
        {draft === "loading" && <p className="mt-2 text-sm" style={{ color: "var(--text-muted)" }}>Generating...</p>}
        {draft === "error" && <p className="mt-2 text-sm" style={{ color: "var(--status-critical)" }}>Failed to generate.</p>}
        {draft && typeof draft === "object" && (
          <div className="mt-2 space-y-2 text-sm">
            <p style={{ color: "var(--text-secondary)" }}>{draft.explanation}</p>
            {draft.draft && (
              <p className="rounded border p-3" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>
                {draft.draft}
              </p>
            )}
            {!draft.draft && (
              <p style={{ color: "var(--text-muted)" }}>No service recommendation available to draft outreach for.</p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
