"""The AI decision assistant (src/ai/graph.py): routes a question to the
structured tools, controlled SQL and RAG layers, gathers their output as
evidence, and only then asks the LLM to answer from that evidence."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.ai.graph import run_query as ai_run_query
from src.ai.observability.tracing import get_trace as ai_get_trace
from src.ai.schemas import AssistantResponse
from src.api.schemas import AssistantQueryRequest
from src.api.security import limit_llm_calls

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/query", response_model=AssistantResponse, dependencies=[Depends(limit_llm_calls)])
def assistant_query(req: AssistantQueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    return ai_run_query(req.query)


@router.get("/trace/{trace_id}")
def assistant_trace(trace_id: str):
    """The trace src/ai/graph.py wrote for one request
    (src/ai/observability/tracing.py). 404 on a miss - there is no
    meaningful partial answer for an unknown trace_id."""
    trace = ai_get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace_id {trace_id!r} not found")
    return trace
