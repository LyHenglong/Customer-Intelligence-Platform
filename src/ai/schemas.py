"""Pydantic schemas for the AI tool layer (src/ai/tools/).

Every tool in src/ai/tools/ returns one of these instead of a raw dict or
DataFrame, so callers - eventually the agent graph in later stages - get a
validated, typed shape rather than whatever a SQL row or joblib artifact
happened to contain.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class Recommendation(BaseModel):
    service: str
    score: float


class CustomerProfile(BaseModel):
    customer_id: str
    contract: Optional[str] = None
    tenure: Optional[int] = None
    monthlycharges: Optional[float] = None
    tenure_bucket: Optional[str] = None
    total_active_services: Optional[int] = None
    customer_satisfaction: Optional[float] = None
    num_complaints: Optional[float] = None


class CustomerLookupResult(BaseModel):
    customer_id: str
    found: bool
    profile: Optional[CustomerProfile] = None
    churn_probability: Optional[float] = None
    churn_threshold: Optional[float] = None
    risk_status: Optional[str] = Field(None, description='"high" or "low", relative to the model threshold')
    model_version: Optional[str] = None
    shap_factors: list[ShapFactor] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    recommender_version: Optional[str] = None


class CustomerSearchFilters(BaseModel):
    min_churn_probability: Optional[float] = None
    max_churn_probability: Optional[float] = None
    min_monthly_charges: Optional[float] = None
    max_monthly_charges: Optional[float] = None
    min_satisfaction: Optional[float] = None
    max_satisfaction: Optional[float] = None
    min_complaints: Optional[float] = None
    contract: Optional[str] = None
    tenure_bucket: Optional[str] = None


class CustomerSearchResult(BaseModel):
    customers: list[CustomerProfile]
    total_matched: int
    limit: int
    offset: int
    truncated: bool


class AggregateBucket(BaseModel):
    key: str
    churn_rate: float
    n_customers: int
    avg_monthly_charges: Optional[float] = None


class AggregateResult(BaseModel):
    dimension: str
    buckets: list[AggregateBucket]


class ChurnAnalysisResult(BaseModel):
    population_size: int
    current_churn_rate: float
    predicted_high_risk_count: int
    mean_churn_probability: float
    median_churn_probability: float
    model_version: Optional[str] = None
    threshold: float
    filters_applied: dict = Field(default_factory=dict)


class RetrainingAnalysisResult(BaseModel):
    current_model_version: Optional[str] = None
    previous_model_version: Optional[str] = None
    current_metrics: Optional[dict] = None
    previous_metrics: Optional[dict] = None
    metric_changes: dict = Field(default_factory=dict)
    latest_drift: list[dict] = Field(default_factory=list)
    retrain_reason: Optional[str] = Field(
        None, description="Latest AI Agent Layer retrain narrative, if one has been generated"
    )
