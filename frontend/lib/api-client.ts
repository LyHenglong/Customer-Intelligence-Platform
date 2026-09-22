// One typed fetch function per src/model/api.py endpoint. NEXT_PUBLIC_API_URL
// is inlined into the browser bundle at BUILD time (Next.js convention for
// NEXT_PUBLIC_* vars) - see docker/Dockerfile.frontend's comment on why
// this means a wrong value needs a rebuild, not a restart.

import type {
  AssistantResponse,
  AtRiskListResponse,
  CustomerLookupResult,
  CustomerSearchFilters,
  CustomerSearchResult,
  ModelHistoryResponse,
  OutreachDraftResponse,
  OverviewStatsResponse,
  PipelineStatusResponse,
  RevenueAtRiskResponse,
  SegmentRatesResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store", // every page here reads live warehouse data, never a stale build-time snapshot
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export function getOverviewStats(threshold?: number) {
  return apiFetch<OverviewStatsResponse>(`/overview/stats${qs({ threshold })}`);
}

export function getSegmentRates(column: string) {
  return apiFetch<SegmentRatesResponse>(`/overview/segment-rates${qs({ column })}`);
}

export function getRevenueAtRiskBySegment(threshold?: number, segment_column?: string) {
  return apiFetch<RevenueAtRiskResponse>(
    `/overview/revenue-at-risk-by-segment${qs({ threshold, segment_column })}`,
  );
}

export function getAtRisk(threshold?: number, max_rows?: number, offset?: number) {
  return apiFetch<AtRiskListResponse>(`/at-risk${qs({ threshold, max_rows, offset })}`);
}

export function postOutreachDraft(customerId: string) {
  return apiFetch<OutreachDraftResponse>(`/outreach-draft/${encodeURIComponent(customerId)}`, {
    method: "POST",
  });
}

export function searchCustomers(filters: CustomerSearchFilters) {
  return apiFetch<CustomerSearchResult>(
    `/customers${qs({
      limit: filters.limit,
      offset: filters.offset,
      min_churn_probability: filters.min_churn_probability,
      max_churn_probability: filters.max_churn_probability,
      min_monthly_charges: filters.min_monthly_charges,
      max_monthly_charges: filters.max_monthly_charges,
      min_satisfaction: filters.min_satisfaction,
      max_satisfaction: filters.max_satisfaction,
      min_complaints: filters.min_complaints,
      contract: filters.contract,
      tenure_bucket: filters.tenure_bucket,
    })}`,
  );
}

export function getCustomer(customerId: string) {
  return apiFetch<CustomerLookupResult>(`/customers/${encodeURIComponent(customerId)}`);
}

export function getModelHistory() {
  return apiFetch<ModelHistoryResponse>("/model-history");
}

export function getPipelineStatus() {
  return apiFetch<PipelineStatusResponse>("/pipeline-status");
}

export function postAssistantQuery(query: string) {
  return apiFetch<AssistantResponse>("/assistant/query", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}
