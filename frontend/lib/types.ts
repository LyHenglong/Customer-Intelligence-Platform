// Mirrors src/model/api.py's response shapes 1:1. Kept as one file since
// the backend keeps all its response models in one module too
// (src/model/api.py itself, plus src/ai/schemas.py for the
// AI/customer-tool shapes) - no reason to split what the source doesn't.

export interface HealthResponse {
  status: string;
  churn_model_loaded: boolean;
  recommender_loaded: boolean;
}

export interface ModelInfoResponse {
  churn_model_version: string | null;
  recommender_version: string | null;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface OverviewStatsResponse {
  total_customers: number;
  historical_churn_rate: number;
  at_risk_count: number;
  revenue_at_risk: number;
  model_auc: number | null;
  model_version: string | null;
  threshold_used: number;
  top_feature_importances: FeatureImportance[];
}

export interface SegmentBucket {
  key: string;
  churn_rate: number;
  n_customers: number;
}

export interface SegmentRatesResponse {
  column: string;
  buckets: SegmentBucket[];
}

export interface RevenueAtRiskBucket {
  segment: string;
  revenue_at_risk: number;
}

export interface RevenueAtRiskResponse {
  segment_column: string;
  buckets: RevenueAtRiskBucket[];
}

export interface ShapFactor {
  feature: string;
  shap_value: number;
  direction: string;
}

export interface AtRiskCustomer {
  customer_id: string;
  churn_probability: number;
  contract: string | null;
  tenure: number | null;
  monthlycharges: number | null;
  key_risk_factors: ShapFactor[];
  recommended_action: string | null;
}

export interface AtRiskListResponse {
  customers: AtRiskCustomer[];
  total_at_risk: number;
  threshold: number;
  max_rows_used: number;
  offset: number;
  model_version: string | null;
}

export interface OutreachDraftResponse {
  customer_id: string;
  explanation: string;
  explanation_source: "llm" | "fallback";
  recommended_service: string | null;
  draft: string | null;
  draft_source: "llm" | "fallback" | null;
  model_version: string;
}

export interface CustomerProfile {
  customer_id: string;
  contract: string | null;
  tenure: number | null;
  monthlycharges: number | null;
  tenure_bucket: string | null;
  total_active_services: number | null;
  customer_satisfaction: number | null;
  num_complaints: number | null;
}

export interface CustomerSearchResult {
  customers: CustomerProfile[];
  total_matched: number;
  limit: number;
  offset: number;
  truncated: boolean;
}

export interface Recommendation {
  service: string;
  score: number;
}

export interface CustomerLookupResult {
  customer_id: string;
  found: boolean;
  profile: CustomerProfile | null;
  churn_probability: number | null;
  churn_threshold: number | null;
  risk_status: "high" | "low" | null;
  model_version: string | null;
  shap_factors: ShapFactor[];
  recommendation: Recommendation[];
  recommender_version: string | null;
}

export interface CustomerSearchFilters {
  min_churn_probability?: number;
  max_churn_probability?: number;
  min_monthly_charges?: number;
  max_monthly_charges?: number;
  min_satisfaction?: number;
  max_satisfaction?: number;
  min_complaints?: number;
  contract?: string;
  tenure_bucket?: string;
  limit?: number;
  offset?: number;
}

// /model-history has no strict backend response_model (see api.py's
// comment on why - metadata shape has drifted across real model
// versions) - kept loose here to match, with the handful of fields every
// page actually reads made optional rather than required.
export interface ModelVersionMetadata {
  version: string;
  trained_at?: string;
  model_type?: string;
  threshold?: number;
  threshold_rationale?: string;
  precision_churn?: number;
  recall_churn?: number;
  f1_churn?: number;
  accuracy?: number;
  roc_auc?: number;
  confusion_matrix?: number[][];
  n_rows_total?: number;
  class_pct?: Record<string, number>;
  [key: string]: unknown;
}

export interface ModelHistoryResponse {
  versions: ModelVersionMetadata[];
}

export interface IngestionLogRow {
  batch_file: string;
  rows_loaded: number;
  loaded_at: string;
  status: string;
}

export interface DriftRow {
  feature: string;
  feature_type: string;
  psi: number;
  severity: string;
  reference_batch: string;
  current_batch: string;
}

export interface RetrainSummary {
  churn_model_version: string;
  previous_version: string | null;
  summary_text: string;
  created_at: string;
}

export interface PipelineStatusResponse {
  ingestion_log: IngestionLogRow[];
  batches_ingested: number;
  total_simulated_batches: number;
  batches_to_next_retrain: number;
  retrain_every_n_batches: number;
  model_versions_trained: number;
  drift: DriftRow[];
  latest_retrain_summary: RetrainSummary | null;
}

export interface Citation {
  label: string;
  type: string;
  source: string;
}

export interface Evidence {
  type: string;
  source: string;
  claim: string;
  value: string | null;
}

export interface AssistantResponse {
  answer: string;
  citations: Citation[];
  evidence: Evidence[];
  tools_used: string[];
  model_version: string | null;
  confidence: number | null;
  route: string | null;
  trace_id: string | null;
  latency_ms: number | null;
}
