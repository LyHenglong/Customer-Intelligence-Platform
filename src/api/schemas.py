"""Request/response models for the serving API (src/api/routers/). The AI
assistant's own schemas live in src/ai/schemas.py and are reused as-is."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    age: Optional[float] = None
    annual_income: Optional[float] = None
    dependents: Optional[float] = None
    tenure: Optional[float] = None
    tenure_years: Optional[float] = None
    monthlycharges: Optional[float] = None
    totalcharges: Optional[float] = None
    num_services: Optional[float] = None
    total_active_services: Optional[float] = None
    cost_per_active_service: Optional[float] = None
    customer_satisfaction: Optional[float] = None
    num_complaints: Optional[float] = None
    num_service_calls: Optional[float] = None
    late_payments: Optional[float] = None
    avg_monthly_gb: Optional[float] = None
    avg_gb_per_service: Optional[float] = None
    days_since_last_interaction: Optional[float] = None
    credit_score: Optional[float] = None
    complaints_per_tenure_month: Optional[float] = None
    annual_spend_to_income_ratio: Optional[float] = None

    senior_citizen: Optional[float] = None
    paperless_billing: Optional[float] = None
    has_phone_service: Optional[float] = None
    has_internet_service: Optional[float] = None
    has_online_security: Optional[float] = None
    has_online_backup: Optional[float] = None
    has_device_protection: Optional[float] = None
    has_tech_support: Optional[float] = None
    has_streaming_tv: Optional[float] = None
    has_streaming_movies: Optional[float] = None
    is_month_to_month: Optional[float] = None
    is_disengaged: Optional[float] = None

    gender: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    contract: Optional[str] = None
    payment_method: Optional[str] = None
    tenure_bucket: Optional[str] = None


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., description="Predicted probability of churn (class 1)")
    churn_prediction: int = Field(..., description="0 = predicted retained, 1 = predicted churn")
    threshold_used: float = Field(..., description="Decision threshold applied to churn_probability")
    model_version: str


class RecommendRequest(BaseModel):
    customer_id: str
    top_n: int = Field(3, ge=1, le=20)


class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: list[dict]
    model_version: str


class ExplainChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_factors: list[dict] = Field(
        ..., description="Raw SHAP attribution: [{feature, shap_value, direction}], always present"
    )
    explanation: str = Field(..., description="Plain-English explanation - AI-generated when available")
    source: str = Field(..., description='"llm" (AI Agent Layer) or "fallback" (raw SHAP text, LLM unavailable)')
    model_version: str


class OutreachDraftResponse(BaseModel):
    customer_id: str
    explanation: str
    explanation_source: str = Field(..., description='"llm" or "fallback"')
    recommended_service: Optional[str] = None
    draft: Optional[str] = Field(None, description="Retention outreach draft - null if no service was recommended")
    draft_source: Optional[str] = Field(None, description='"llm" or "fallback", null iff draft is null')
    model_version: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class OverviewStatsResponse(BaseModel):
    total_customers: int
    historical_churn_rate: float
    at_risk_count: int
    revenue_at_risk: float
    model_auc: Optional[float] = None
    model_version: Optional[str] = None
    threshold_used: float
    top_feature_importances: list[FeatureImportance] = Field(default_factory=list)


class SegmentBucket(BaseModel):
    key: str
    churn_rate: float
    n_customers: int


class SegmentRatesResponse(BaseModel):
    column: str
    buckets: list[SegmentBucket]


class RevenueAtRiskBucket(BaseModel):
    segment: str
    revenue_at_risk: float


class RevenueAtRiskResponse(BaseModel):
    segment_column: str
    buckets: list[RevenueAtRiskBucket]


class AtRiskCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    contract: Optional[str] = None
    tenure: Optional[int] = None
    monthlycharges: Optional[float] = None
    key_risk_factors: list[dict] = Field(default_factory=list)
    recommended_action: Optional[str] = None


class AtRiskListResponse(BaseModel):
    customers: list[AtRiskCustomer]
    total_at_risk: int
    threshold: float
    max_rows_used: int
    offset: int
    model_version: Optional[str] = None


class AssistantQueryRequest(BaseModel):
    # Bounded: every character is sent to a paid LLM, and a real question
    # about this data never needs more.
    query: str = Field(..., max_length=2000)
    conversation_id: Optional[str] = None  # accepted, not yet used - see docs/ai_layer_build_plan.md section 27
