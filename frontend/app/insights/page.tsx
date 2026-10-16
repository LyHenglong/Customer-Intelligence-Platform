"use client";

import { useEffect, useState } from "react";
import {
  getOverviewStats,
  getSegmentRates,
  getRevenueAtRiskBySegment,
  getPipelineStatus,
  ApiError,
} from "@/lib/api-client";
import type { OverviewStatsResponse, SegmentRatesResponse, RevenueAtRiskResponse, PipelineStatusResponse } from "@/lib/types";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import LoadingState from "@/components/LoadingState";
import ErrorState from "@/components/ErrorState";

interface Insight {
  id: string;
  headline: string;
  metric: string;
  implication: string;
  action: string;
  tone: "critical" | "warning" | "info";
}

const TONE_COLOR: Record<Insight["tone"], string> = {
  critical: "var(--status-critical)",
  warning: "var(--status-warning)",
  info: "var(--series-2)",
};

/**
 * Every insight here is computed from numbers the app already fetches
 * elsewhere (overview stats, segment rates, revenue-at-risk, drift) - none
 * are an LLM call and none are invented. The "implication"/"action" text
 * is a template keyed off which real comparison fired, not free text, so
 * every sentence traces back to a field in lib/types.ts.
 */
function buildInsights(
  stats: OverviewStatsResponse,
  contractRates: SegmentRatesResponse,
  tenureRates: SegmentRatesResponse,
  revenueAtRisk: RevenueAtRiskResponse,
  pipeline: PipelineStatusResponse,
): Insight[] {
  const insights: Insight[] = [];

  // 1. Contract churn spread
  if (contractRates.buckets.length >= 2) {
    const sorted = [...contractRates.buckets].sort((a, b) => b.churn_rate - a.churn_rate);
    const worst = sorted[0];
    const best = sorted[sorted.length - 1];
    if (best.churn_rate > 0) {
      const ratio = worst.churn_rate / best.churn_rate;
      insights.push({
        id: "contract-spread",
        headline: `${worst.key.replaceAll("_", " ")} customers churn ${ratio.toFixed(1)}x the rate of ${best.key.replaceAll("_", " ")} customers`,
        metric: `${(worst.churn_rate * 100).toFixed(1)}% (${worst.n_customers.toLocaleString()} customers) vs ${(best.churn_rate * 100).toFixed(1)}% (${best.n_customers.toLocaleString()} customers)`,
        implication: "Contract type is one of the strongest churn signals available - shorter commitments carry far more risk than longer ones.",
        action: `Prioritize contract-upgrade offers for ${worst.key.replaceAll("_", " ")} customers before they reach a renewal decision.`,
        tone: ratio >= 2 ? "critical" : "warning",
      });
    }
  }

  // 2. Tenure churn spread
  if (tenureRates.buckets.length >= 2) {
    const sorted = [...tenureRates.buckets].sort((a, b) => b.churn_rate - a.churn_rate);
    const worst = sorted[0];
    const best = sorted[sorted.length - 1];
    if (worst.key !== best.key) {
      insights.push({
        id: "tenure-spread",
        headline: `Customers in the "${worst.key.replaceAll("_", " ")}" tenure bucket churn at the highest rate`,
        metric: `${(worst.churn_rate * 100).toFixed(1)}% churn across ${worst.n_customers.toLocaleString()} customers`,
        implication: "Churn concentrates early in the customer lifecycle - customers who make it past this stage are meaningfully stickier.",
        action: "Focus onboarding and early-engagement outreach on customers still in this tenure window.",
        tone: "warning",
      });
    }
  }

  // 3. Revenue concentration
  if (revenueAtRisk.buckets.length > 0 && stats.revenue_at_risk > 0) {
    const top = [...revenueAtRisk.buckets].sort((a, b) => b.revenue_at_risk - a.revenue_at_risk)[0];
    const share = top.revenue_at_risk / stats.revenue_at_risk;
    if (share >= 0.3) {
      insights.push({
        id: "revenue-concentration",
        headline: `${(share * 100).toFixed(0)}% of at-risk monthly revenue sits in the ${top.segment.replaceAll("_", " ")} segment`,
        metric: `$${top.revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })} of $${stats.revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })} at-risk revenue/mo`,
        implication: "Retention spend concentrated on this one segment would cover a disproportionate share of the total revenue exposure.",
        action: `Build a dedicated retention offer for the ${top.segment.replaceAll("_", " ")} segment before spreading budget thinner across others.`,
        tone: "critical",
      });
    }
  }

  // 4. Feature drift
  const significant = pipeline.drift.filter((d) => d.severity === "significant");
  if (significant.length > 0) {
    insights.push({
      id: "drift",
      headline: `${significant.length} feature${significant.length > 1 ? "s have" : " has"} drifted significantly since the reference batch`,
      metric: significant.map((d) => `${d.feature.replaceAll("_", " ")} (PSI ${d.psi.toFixed(3)})`).join(", "),
      implication: "The population the model now sees differs meaningfully from what it was trained on, which can silently erode prediction quality.",
      action: "Check whether a retrain is due on Pipeline Status, and treat this model's predictions with more caution until it runs.",
      tone: "critical",
    });
  }

  // 5. Top churn driver
  if (stats.top_feature_importances.length > 0) {
    const top = stats.top_feature_importances[0];
    insights.push({
      id: "top-driver",
      headline: `${top.feature.replaceAll("_", " ")} is the single biggest churn driver the model has learned`,
      metric: `${(top.importance * 100).toFixed(0)}% of total feature importance`,
      implication: "Changes here move churn risk more than any other factor the model tracks.",
      action: `When triaging at-risk customers, check their ${top.feature.replaceAll("_", " ")} first - it is the highest-leverage signal available.`,
      tone: "info",
    });
  }

  // 6. At-risk exposure
  if (stats.total_customers > 0) {
    const share = stats.at_risk_count / stats.total_customers;
    insights.push({
      id: "at-risk-share",
      headline: `${(share * 100).toFixed(1)}% of the customer base is above the churn-risk threshold`,
      metric: `${stats.at_risk_count.toLocaleString()} of ${stats.total_customers.toLocaleString()} customers at threshold ${stats.threshold_used.toFixed(2)}, representing $${stats.revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo`,
      implication: "This is the pool retention efforts are currently sized against - raising or lowering the threshold changes both the target list and the workload.",
      action: "Review the At-risk customers list and Recommendations to work this pool by priority.",
      tone: share >= 0.2 ? "critical" : "info",
    });
  }

  return insights;
}

export default function InsightsPage() {
  const [insights, setInsights] = useState<Insight[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      getOverviewStats(),
      getSegmentRates("contract"),
      getSegmentRates("tenure_bucket"),
      getRevenueAtRiskBySegment(),
      getPipelineStatus(),
    ])
      .then(([stats, contractRates, tenureRates, revenueAtRisk, pipeline]) => {
        if (cancelled) return;
        setInsights(buildInsights(stats, contractRates, tenureRates, revenueAtRisk, pipeline));
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e.message : "Failed to load insights.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <ErrorState message={error} />;
  if (loading || !insights) return <LoadingState />;

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Customer Intelligence"
        title="Insights"
        description="Automatically generated observations from the current customer, revenue, and model data - not predictions, just patterns worth acting on."
      />

      {insights.length === 0 ? (
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Nothing notable found in the current data.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {insights.map((insight) => (
            <Card key={insight.id} className="relative overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 w-1"
                style={{ background: TONE_COLOR[insight.tone] }}
                aria-hidden="true"
              />
              <div className="pl-2">
                <h3 className="text-[14.5px] font-semibold leading-snug" style={{ color: "var(--text-primary)" }}>
                  {insight.headline}
                </h3>
                <p className="mt-1.5 text-[12px] tabular-nums" style={{ color: "var(--text-muted)" }}>
                  {insight.metric}
                </p>
                <p className="mt-2.5 text-[12.5px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                  {insight.implication}
                </p>
                <div
                  className="mt-3 rounded-lg px-3 py-2 text-[12px] font-medium"
                  style={{ background: "var(--brand-tint)", color: "var(--brand-strong)" }}
                >
                  Suggested action: {insight.action}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
