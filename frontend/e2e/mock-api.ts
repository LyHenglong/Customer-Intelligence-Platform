import type { Page, Route } from "@playwright/test";
import type {
  AssistantResponse,
  AtRiskListResponse,
  CustomerLookupResult,
  CustomerSearchResult,
  ModelHistoryResponse,
  OutreachDraftResponse,
  OverviewStatsResponse,
  PipelineStatusResponse,
  RevenueAtRiskResponse,
  SegmentRatesResponse,
} from "../lib/types";

// Fixture responses typed against lib/types.ts, so a backend contract
// change that the UI types pick up also breaks these fixtures at compile time.

const profile = {
  customer_id: "CUST0001",
  contract: "month-to-month",
  tenure: 4,
  monthlycharges: 89.5,
  tenure_bucket: "new_0_6mo",
  total_active_services: 3,
  customer_satisfaction: 2,
  num_complaints: 3,
};

const shap = [
  { feature: "contract", shap_value: 0.42, direction: "increases risk" },
  { feature: "tenure", shap_value: 0.21, direction: "increases risk" },
];

export const fixtures = {
  overviewStats: {
    total_customers: 1_000_000,
    historical_churn_rate: 0.0992,
    at_risk_count: 48_210,
    revenue_at_risk: 3_912_004.5,
    model_auc: 0.6661,
    model_version: "20260923T054451Z",
    threshold_used: 0.35,
    top_feature_importances: [
      { feature: "contract", importance: 0.31 },
      { feature: "tenure", importance: 0.22 },
    ],
  } satisfies OverviewStatsResponse,
  segmentRates: (column: string): SegmentRatesResponse => ({
    column,
    buckets: [
      { key: "a", churn_rate: 0.15, n_customers: 400_000 },
      { key: "b", churn_rate: 0.05, n_customers: 600_000 },
    ],
  }),
  revenueAtRisk: {
    segment_column: "contract",
    buckets: [
      { segment: "month-to-month", revenue_at_risk: 2_500_000 },
      { segment: "two_year", revenue_at_risk: 400_000 },
    ],
  } satisfies RevenueAtRiskResponse,
  atRisk: {
    customers: [
      {
        customer_id: "CUST0001",
        churn_probability: 0.91,
        contract: "month-to-month",
        tenure: 4,
        monthlycharges: 89.5,
        key_risk_factors: shap,
        recommended_action: "has_tech_support",
      },
    ],
    total_at_risk: 1,
    threshold: 0.35,
    max_rows_used: 25,
    offset: 0,
    model_version: "20260923T054451Z",
  } satisfies AtRiskListResponse,
  outreach: {
    customer_id: "CUST0001",
    explanation: "Short tenure on a month-to-month contract drives this customer's risk.",
    explanation_source: "llm",
    recommended_service: "has_tech_support",
    draft: "We'd love to offer you free tech support for three months.",
    draft_source: "llm",
    model_version: "20260923T054451Z",
  } satisfies OutreachDraftResponse,
  customers: {
    customers: [profile],
    total_matched: 1,
    limit: 25,
    offset: 0,
    truncated: false,
  } satisfies CustomerSearchResult,
  customer: {
    customer_id: "CUST0001",
    found: true,
    profile,
    churn_probability: 0.91,
    churn_threshold: 0.35,
    risk_status: "high",
    model_version: "20260923T054451Z",
    shap_factors: shap,
    recommendation: [{ service: "has_tech_support", score: 0.8 }],
    recommender_version: "20260921T040152Z",
  } satisfies CustomerLookupResult,
  modelHistory: {
    versions: [
      {
        version: "20260923T054451Z",
        trained_at: "2026-09-23T05:44:51Z",
        model_type: "LightGBM",
        threshold: 0.35,
        precision_churn: 0.18,
        recall_churn: 0.6,
        f1_churn: 0.28,
        accuracy: 0.7,
        roc_auc: 0.6661,
        confusion_matrix: [
          [100, 20],
          [5, 10],
        ],
      },
    ],
    serving_version: "20260923T054451Z",
  } satisfies ModelHistoryResponse,
  pipelineStatus: {
    ingestion_log: [{ batch_file: "batch_001.csv", rows_loaded: 76_924, loaded_at: "2026-09-09T07:00:00", status: "success" }],
    batches_ingested: 1,
    total_simulated_batches: 13,
    batches_to_next_retrain: 2,
    retrain_every_n_batches: 3,
    model_versions_trained: 1,
    drift: [
      {
        feature: "tenure",
        feature_type: "numeric",
        psi: 0.01,
        severity: "none",
        reference_batch: "batch_001.csv",
        current_batch: "batch_002.csv",
      },
    ],
    latest_retrain_summary: {
      churn_model_version: "20260923T054451Z",
      previous_version: null,
      summary_text: "The retrained model improved AUC slightly.",
      created_at: "2026-09-23T06:00:00",
    },
  } satisfies PipelineStatusResponse,
  assistant: {
    answer: "Month-to-month customers churn about three times as often as two-year customers.",
    citations: [{ label: "[1]", type: "sql", source: "marts.customer_360" }],
    evidence: [{ type: "sql", source: "marts.customer_360", claim: "churn rate by contract", value: "0.15 vs 0.05" }],
    tools_used: ["sql_analysis"],
    model_version: "20260923T054451Z",
    confidence: 0.9,
    route: "SQL_ANALYSIS",
    trace_id: "trace-123",
    latency_ms: 850,
  } satisfies AssistantResponse,
};

function json(route: Route, body: unknown) {
  return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
}

/** Answers every /api/backend/* request from the fixtures above. An
 *  unexpected path fails loudly rather than hanging on a real network call. */
export async function mockApi(page: Page) {
  await page.route("**/api/backend/**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace(/^\/api\/backend\//, "");

    if (path === "overview/stats") return json(route, fixtures.overviewStats);
    if (path === "overview/segment-rates") return json(route, fixtures.segmentRates(url.searchParams.get("column") ?? ""));
    if (path === "overview/revenue-at-risk-by-segment") return json(route, fixtures.revenueAtRisk);
    if (path === "at-risk") return json(route, fixtures.atRisk);
    if (path.startsWith("outreach-draft/")) return json(route, fixtures.outreach);
    if (path === "customers") return json(route, fixtures.customers);
    if (path.startsWith("customers/")) return json(route, fixtures.customer);
    if (path === "model-history") return json(route, fixtures.modelHistory);
    if (path === "pipeline-status") return json(route, fixtures.pipelineStatus);
    if (path === "assistant/query") return json(route, fixtures.assistant);

    return route.fulfill({ status: 500, body: `unmocked API path in test: ${path}` });
  });
}
