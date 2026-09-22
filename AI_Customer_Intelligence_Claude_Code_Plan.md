# Claude Code Build Plan

## AI-Powered Customer Intelligence & Decision Platform

## 1. Mission

Upgrade the existing **Telecom Customer Churn & Recommendation
Platform** into a production-oriented **AI-Powered Customer Intelligence
& Decision Platform**.

**Do not rebuild the existing project from scratch.** Preserve the
current data platform, churn model, recommender, monitoring, API,
dashboard, tests, Docker setup, and Airflow/dbt architecture. Add a new
AI/analytics layer on top of them.

The existing system already provides: - 1M-row synthetic telecom
customer dataset - DuckDB ingestion and quality gates - PostgreSQL
warehouse - dbt staging/intermediate/customer_360 mart - LightGBM churn
model with probability calibration - content-based recommender - SHAP
explanations - PSI drift monitoring - conditional retraining through
Airflow - FastAPI serving - Streamlit executive dashboard - Docker
Compose - CI with pytest/dbt tests - an existing presentation-only Groq
agent layer

The README describes this existing architecture and verified behavior.
Do not remove working functionality merely to introduce the new layer.

------------------------------------------------------------------------

# 2. New Product Goal

Turn the platform from:

> "A dashboard that shows churn predictions and recommendations"

into:

> "An AI decision assistant that can investigate customer/business
> questions using structured data, ML outputs, and trusted business
> knowledge, then return evidence-backed answers."

Example questions:

-   "Why is churn increasing?"
-   "Which customer segments are driving the increase?"
-   "What are the biggest risk factors?"
-   "Which high-value customers are most at risk?"
-   "What services should we consider offering them?"
-   "Compare churn between month-to-month and long-term customers."
-   "Show me customers with high churn probability and low
    satisfaction."
-   "What changed after the latest model retraining?"
-   "What retention actions are supported by our business
    documentation?"
-   "Why should I trust this answer?"

The assistant must **analyze first and generate language second**.

The LLM must not invent metrics, customer attributes, model predictions,
recommendations, or business facts.

------------------------------------------------------------------------

# 3. Core Design Principle

The new architecture must follow this rule:

``` text
DATA / ML SYSTEM
      |
      | authoritative facts
      v
ANALYSIS TOOLS
      |
      | structured results
      v
AGENT / ORCHESTRATOR
      |
      | grounded evidence
      v
LLM RESPONSE
      |
      v
CITATIONS + TRACE + GUARDRAILS
```

The LLM is an orchestrator and explanation layer.

It is NOT the source of truth.

Authoritative sources:

1.  PostgreSQL/customer_360 for customer/business data
2.  ML artifacts for churn predictions
3.  recommender artifact for recommendations
4.  SHAP for model explanations
5.  drift/retraining tables for model monitoring
6.  RAG corpus for external/business documents

------------------------------------------------------------------------

# 4. Do Not Overengineer

Do not introduce every modern technology just because it is available.

Avoid initially: - Kubernetes - Kafka - Celery unless genuinely
required - multiple independent microservices - multiple LLM agents that
duplicate each other - fine-tuning a large LLM - unnecessary cloud
infrastructure - arbitrary vector databases - LangChain abstractions
where plain Python is clearer

The first implementation should be a clean modular monolith around the
existing FastAPI application.

Use an agent graph only where it adds real routing/tool-use value.

------------------------------------------------------------------------

# 5. Target Architecture

``` text
                           ┌─────────────────────┐
                           │  Streamlit / Web UI  │
                           └──────────┬──────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │    FastAPI    │
                              └───────┬───────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   Query Router      │
                           │    LangGraph        │
                           └──────────┬──────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
      │ SQL Analyst  │        │ RAG Retriever│        │ ML Analyst   │
      │ PostgreSQL   │        │ Vector+BM25  │        │ Churn/SHAP   │
      └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                           ┌─────────────────────┐
                           │ Evidence Aggregator │
                           └──────────┬──────────┘
                                      ▼
                           ┌─────────────────────┐
                           │ Grounding / Safety  │
                           │    Validation       │
                           └──────────┬──────────┘
                                      ▼
                           ┌─────────────────────┐
                           │       LLM           │
                           │ Structured output   │
                           └──────────┬──────────┘
                                      ▼
                           ┌─────────────────────┐
                           │ Answer + Evidence   │
                           │ + Citations + Trace │
                           └─────────────────────┘
```

------------------------------------------------------------------------

# 6. Repository Rules

Before modifying code:

1.  Inspect the complete repository.
2.  Read existing modules before creating replacements.
3.  Identify current interfaces and dependencies.
4.  Run the existing test suite.
5.  Run the existing application if possible.
6.  Create a short implementation inventory documenting:
    -   existing API endpoints
    -   existing database tables
    -   existing model artifact format
    -   existing agent interfaces
    -   existing dashboard pages
    -   existing Docker services
    -   existing environment variables
7.  Preserve backward compatibility.

Do not rewrite functioning modules without a concrete reason.

------------------------------------------------------------------------

# 7. New Directory Structure

Add only what is necessary.

Suggested structure:

``` text
src/
├── ai/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── router.py
│   ├── graph.py
│   ├── llm.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── sql_tool.py
│   │   ├── customer_tool.py
│   │   ├── churn_tool.py
│   │   ├── recommender_tool.py
│   │   ├── shap_tool.py
│   │   ├── aggregate_tool.py
│   │   └── retraining_tool.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── bm25.py
│   │   ├── vector_store.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── guardrails/
│   │   ├── grounding.py
│   │   ├── validation.py
│   │   ├── sql_safety.py
│   │   └── citation_validation.py
│   │
│   ├── evaluation/
│   │   ├── datasets.py
│   │   ├── retrieval.py
│   │   ├── answer.py
│   │   ├── benchmark.py
│   │   └── regression.py
│   │
│   └── observability/
│       ├── tracing.py
│       ├── metrics.py
│       └── cost.py
│
├── ...
```

Do not duplicate existing functionality. Reuse existing `src/model`,
`src/monitoring`, `src/agents`, and `src/warehouse` wherever
appropriate.

------------------------------------------------------------------------

# 8. Phase 1 --- Structured Analytics Tool Layer

Build the structured-data tool layer first.

The agent needs reliable tools rather than direct unrestricted database
access.

## Required tools

### `customer_lookup`

Input: - customer_id

Output: - customer profile - churn probability - churn threshold - risk
status - recommendation - relevant SHAP factors

### `customer_search`

Filters such as: - churn probability - monthly charges - satisfaction -
contract - tenure - segment - service - complaints

Must support pagination and a maximum result limit.

### `aggregate_analysis`

Examples: - churn rate by contract - churn by customer segment - average
monthly charges - churn by tenure bucket - service adoption -
revenue/value summaries

### `churn_analysis`

Return: - current churn rate - selected population - predicted high-risk
count - probability statistics - model version - threshold

### `customer_explanation`

Reuse the existing SHAP implementation.

Do not ask the LLM to calculate SHAP.

### `recommendation_analysis`

Reuse the existing recommender.

The agent must receive the recommendation produced by the recommender
rather than generating its own.

### `retraining_analysis`

Expose: - current model version - previous model version - model
metrics - drift metrics - retraining reason - metric changes

------------------------------------------------------------------------

# 9. SQL Agent Safety

Do NOT give the LLM unrestricted PostgreSQL credentials.

Implement a controlled SQL tool.

Requirements:

-   read-only connection
-   SELECT-only
-   reject INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE
-   reject multiple statements
-   statement timeout
-   row limit
-   query validation
-   allowed-schema/table policy
-   parameterized values where applicable
-   log generated SQL
-   log execution time
-   return structured results

Prefer querying approved views/marts such as:

``` text
marts.customer_360
public.feature_drift
public.ingestion_log
public.retrain_summaries
```

Do not expose raw credentials to the LLM.

------------------------------------------------------------------------

# 10. Phase 2 --- RAG Knowledge Layer

Add a genuine RAG system.

The purpose is NOT to upload arbitrary PDFs and call it ChatPDF.

The corpus should contain business/telecom knowledge relevant to
interpreting the customer data, such as:

``` text
knowledge/
├── retention/
├── telecom_services/
├── customer_support/
├── churn_strategy/
├── product_catalog/
└── policies/
```

Use realistic documents with metadata.

Each chunk should store:

``` text
document_id
title
source
section
page
chunk_id
text
created_at
metadata
```

The assistant must be able to cite the exact source document and
relevant passage.

------------------------------------------------------------------------

# 11. RAG Pipeline

Implement:

``` text
Document
   ↓
Parsing
   ↓
Cleaning
   ↓
Structure-aware chunking
   ↓
Embedding
   ↓
Vector index
   +
BM25 index
   ↓
Hybrid retrieval
   ↓
Candidate fusion
   ↓
Cross-encoder reranking
   ↓
Top context
```

Do not blindly choose one chunk size.

Implement configurable chunking and benchmark at least:

-   fixed-size chunks
-   overlapping chunks
-   structure-aware chunks

Record retrieval performance.

------------------------------------------------------------------------

# 12. Hybrid Retrieval

Implement two retrieval paths:

### Dense

Embedding similarity.

### Sparse

BM25 keyword retrieval.

Combine the rankings using a documented fusion strategy such as
Reciprocal Rank Fusion.

Then rerank the candidate pool using a cross-encoder.

Example:

``` text
BM25 top 20
+
Vector top 20
      ↓
RRF
      ↓
Top 20
      ↓
Cross Encoder
      ↓
Top 5
```

Return retrieval metadata:

``` json
{
  "document_id": "...",
  "chunk_id": "...",
  "score": 0.82,
  "rank": 1,
  "retrieval_method": "hybrid_reranked"
}
```

------------------------------------------------------------------------

# 13. Phase 3 --- Agent Workflow

Use LangGraph or an equivalent explicit state-machine implementation.

Do not build a fake multi-agent system where every node simply calls the
same LLM.

Use meaningful roles:

``` text
START
  ↓
Query Classification
  ↓
Planner / Router
  ├── SQL Analysis
  ├── Customer/ML Analysis
  ├── RAG Retrieval
  └── Combined Analysis
          ↓
Evidence Aggregation
          ↓
Evidence Sufficiency Check
      ┌───┴────┐
      │        │
   enough    insufficient
      │        │
      ▼        ▼
 Generate   Retrieve/Analyze again
      │
      ▼
 Validate
      │
      ▼
 Answer
```

The system should support multi-tool questions.

Example:

> "Why did churn increase among high-value customers, and what retention
> actions do our policies recommend?"

Workflow:

``` text
SQL
 ↓
identify high-value population
 ↓
churn analysis
 ↓
SHAP/ML analysis
 ↓
RAG policy retrieval
 ↓
combine evidence
 ↓
generate answer
 ↓
validate citations
```

------------------------------------------------------------------------

# 14. Query Routing

Classify queries into:

``` text
CUSTOMER_LOOKUP
SQL_ANALYSIS
ML_ANALYSIS
RAG_SEARCH
MULTI_SOURCE
UNSUPPORTED
```

Examples:

  Query                                         Route
  --------------------------------------------- -------------
  "Show customer C123"                          Customer
  "Why is churn higher for month-to-month?"     SQL + ML
  "What are this customer's risk factors?"      ML/SHAP
  "What does our retention policy say?"         RAG
  "Which customers should we target?"           SQL + ML
  "What policy supports this recommendation?"   ML + RAG
  "Tell me tomorrow's weather"                  Unsupported

Unsupported questions should receive a controlled response rather than
hallucinating.

------------------------------------------------------------------------

# 15. Phase 4 --- Evidence-Based Answer Generation

Every final answer should have structured internal evidence.

Example internal object:

``` json
{
  "answer": "...",
  "evidence": [
    {
      "type": "database",
      "source": "marts.customer_360",
      "claim": "...",
      "value": "..."
    },
    {
      "type": "model",
      "model_version": "...",
      "claim": "...",
      "value": "..."
    },
    {
      "type": "document",
      "document_id": "...",
      "chunk_id": "...",
      "claim": "..."
    }
  ],
  "confidence": 0.84,
  "citations": [...]
}
```

Do not allow unsupported numerical claims.

If the evidence does not contain enough information:

> "I don't have enough evidence in the available data to answer that
> reliably."

This is preferable to guessing.

------------------------------------------------------------------------

# 16. Citation System

Citations must be first-class objects.

For database-derived claims:

``` text
[Customer 360: 1,000,000 customer records]
```

For RAG:

``` text
[Retention Policy, section 3.2, page 7]
```

For model results:

``` text
[Churn Model v2026..., ROC-AUC 0.xxx]
```

The UI should let users inspect the evidence behind a claim.

For RAG citations, display: - document title - section - page if
available - exact supporting passage - retrieval score

Do not fabricate page numbers or document metadata.

------------------------------------------------------------------------

# 17. Guardrails

Implement at least these guardrails:

## Hallucination prevention

-   answer only from tool results
-   reject unsupported numerical claims
-   require evidence for factual assertions
-   require citations for RAG-derived claims
-   allow explicit "insufficient evidence"

## SQL safety

-   read-only
-   SELECT only
-   timeout
-   row limit
-   allowed tables
-   query logging

## Tool safety

-   validate tool inputs
-   enforce maximum result sizes
-   prevent arbitrary filesystem access
-   prevent arbitrary shell execution

## Output safety

Use Pydantic schemas.

Example:

``` python
class AssistantResponse(BaseModel):
    answer: str
    citations: list[Citation]
    evidence: list[Evidence]
    tools_used: list[str]
    model_version: str | None
    confidence: float | None
```

Do not return arbitrary LLM JSON directly to the frontend.

------------------------------------------------------------------------

# 18. Phase 5 --- Evaluation Harness

This is mandatory.

Do not evaluate the assistant with five manually chosen questions.

Create an evaluation dataset of at least **100 questions**.

Prefer 150--200 if practical.

Categories:

``` text
30 SQL/data questions
20 customer-specific questions
20 ML/churn questions
20 RAG questions
20 multi-source questions
10 unsupported/adversarial questions
```

Each test case should include:

``` json
{
  "id": "Q001",
  "question": "...",
  "expected_tools": ["sql", "ml"],
  "expected_sources": ["customer_360"],
  "expected_facts": ["..."],
  "expected_documents": [],
  "difficulty": "medium"
}
```

------------------------------------------------------------------------

# 19. Retrieval Metrics

Measure:

-   Recall@K
-   Precision@K
-   Hit@K
-   MRR
-   nDCG where appropriate

Compare:

``` text
Vector only
BM25 only
Hybrid
Hybrid + reranker
```

Also benchmark chunking strategies.

Record:

``` text
retrieval_quality
latency_ms
embedding_latency
reranker_latency
```

The README should contain real benchmark results generated by the
project.

Never invent benchmark numbers.

------------------------------------------------------------------------

# 20. Answer Evaluation

Evaluate:

-   faithfulness
-   answer relevancy
-   context precision
-   context recall
-   citation correctness
-   unsupported-claim rate
-   tool-selection accuracy

Use RAGAS/DeepEval or equivalent where appropriate, but do not blindly
trust one automated metric.

Include a small manually reviewed golden set.

------------------------------------------------------------------------

# 21. Regression Evaluation

Every important change should be able to run against the benchmark.

Create a command such as:

``` bash
python -m src.ai.evaluation.benchmark
```

Output:

``` text
=====================================
AI ASSISTANT EVALUATION
=====================================

Queries:                 150
Tool selection accuracy: 94.7%
Retrieval Recall@5:      91.3%
MRR:                     0.84
Faithfulness:            0.91
Answer relevancy:        0.89
Citation accuracy:       96.1%
Unsupported claims:      1.3%
Median latency:          1.82s
P95 latency:             4.91s
=====================================
```

Numbers must be calculated from actual runs.

------------------------------------------------------------------------

# 22. Phase 6 --- Observability

Every assistant request should produce a trace.

Track:

``` text
request_id
timestamp
user_query
route
tools_used
tool_latency
SQL query hash
retrieval latency
retrieved documents
reranker latency
LLM model
input tokens
output tokens
estimated cost
total latency
validation result
fallback status
error
```

Do not store sensitive customer data unnecessarily.

Create an internal observability view.

Minimum dashboard metrics:

-   requests
-   success rate
-   error rate
-   average latency
-   p50 latency
-   p95 latency
-   tool usage
-   retrieval hit rate
-   fallback rate
-   token usage
-   estimated cost
-   unsupported-query rate

------------------------------------------------------------------------

# 23. Phase 7 --- LLM Provider Abstraction

Do not tightly couple the application to one provider.

Create:

``` text
LLMProvider
    ├── GroqProvider
    ├── OpenAICompatibleProvider
    └── LocalProvider (optional)
```

The application should depend on the interface.

Configuration should come from environment variables.

Example:

``` text
LLM_PROVIDER=groq
LLM_MODEL=...
EMBEDDING_MODEL=...
RERANKER_MODEL=...
```

Keep the existing Groq retry/backoff/fallback behavior.

The existing project already handles API failure, caching, rate limits,
and graceful fallback; extend rather than duplicate those patterns.

------------------------------------------------------------------------

# 24. Phase 8 --- Model/AI Experiment Tracking

Use MLflow where it provides actual value.

Track:

### ML

-   model version
-   training dataset size
-   features
-   hyperparameters
-   ROC-AUC
-   precision
-   recall
-   F1
-   Brier score
-   threshold

### RAG

-   embedding model
-   chunking strategy
-   chunk size
-   overlap
-   retrieval method
-   reranker
-   Recall@K
-   MRR
-   latency

### LLM

-   model
-   prompt version
-   token usage
-   latency
-   evaluation metrics

Do not introduce MLflow simply as a decorative dependency.

------------------------------------------------------------------------

# 25. Phase 9 --- Dashboard Upgrade

Keep the existing dashboard views.

Add a new:

## AI Decision Assistant

Example UI:

``` text
┌───────────────────────────────────────────────────┐
│ AI Customer Intelligence Assistant               │
├───────────────────────────────────────────────────┤
│ Ask a business question...                        │
│                                                   │
│ Why did churn increase among high-value users?   │
│                                                   │
│ [ Analyze ]                                       │
└───────────────────────────────────────────────────┘
```

Response:

``` text
Churn among high-value customers increased from X%
to Y% in the latest available period.

The increase is concentrated in ...

Key evidence
────────────────────────
• Contract type
• Satisfaction
• Complaints
• Service usage

Model evidence
────────────────────────
...

Business guidance
────────────────────────
...

Sources
────────────────────────
[1] Customer 360
[2] Churn Model v...
[3] Retention Policy §3.2

Tools used
────────────────────────
SQL Analysis
Churn Analysis
SHAP
RAG
```

Allow the user to expand evidence.

------------------------------------------------------------------------

# 26. Customer-Level Assistant

Add an assistant to the existing At-Risk Customers experience.

For a selected customer:

``` text
Customer CUST000...

Churn probability: 32.6%
Risk status: High

Why?
1. ...
2. ...
3. ...

Recommended service:
Internet Service

Why this recommendation?
...

Relevant business guidance:
...

Draft outreach:
...
```

The recommendation and SHAP values must come from the existing systems.

The LLM only explains them.

------------------------------------------------------------------------

# 27. Conversation Memory

Do not implement unrestricted long-term memory initially.

For a single conversation, maintain:

``` text
conversation_id
previous questions
tool results
relevant evidence
```

Limit history size.

Do not store unnecessary customer information.

If persistent conversation history is later added, define retention
rules and privacy boundaries first.

------------------------------------------------------------------------

# 28. API

Add a new endpoint:

``` http
POST /assistant/query
```

Request:

``` json
{
  "query": "Why did churn increase among high-value customers?",
  "conversation_id": "optional-id"
}
```

Response:

``` json
{
  "answer": "...",
  "citations": [],
  "evidence": [],
  "tools_used": [
    "aggregate_analysis",
    "churn_analysis",
    "shap_analysis",
    "rag_search"
  ],
  "trace_id": "...",
  "latency_ms": 1830
}
```

Also add:

``` http
GET /assistant/trace/{trace_id}
```

for debugging/observability.

If streaming is useful, add:

``` http
POST /assistant/query/stream
```

after the non-streaming endpoint is stable.

------------------------------------------------------------------------

# 29. Testing

Add tests for:

### Tools

-   valid customer lookup
-   missing customer
-   filters
-   pagination
-   SQL safety
-   query timeout
-   row limits

### RAG

-   document ingestion
-   chunk metadata
-   BM25
-   vector retrieval
-   hybrid fusion
-   reranking
-   citation metadata

### Agent

-   routing
-   tool selection
-   multi-tool workflows
-   insufficient evidence
-   unsupported questions

### Guardrails

-   hallucinated number rejection
-   invalid citation rejection
-   unsafe SQL rejection
-   malformed LLM output
-   provider failure

### Observability

-   trace creation
-   latency logging
-   token/cost tracking

### API

-   successful query
-   validation error
-   provider failure
-   timeout
-   fallback

All external LLM calls must be mocked in CI.

------------------------------------------------------------------------

# 30. CI Requirements

Existing tests must continue to pass.

Add:

``` text
pytest
```

for the new AI components.

CI must not require: - live Groq key - live OpenAI key - live vector
database - live external website

Use deterministic fixtures and mocks.

Add a lightweight evaluation regression suite to CI.

The full 150--200 question benchmark can remain a separate evaluation
command if it is too expensive for every commit.

------------------------------------------------------------------------

# 31. Docker

Extend the existing Docker setup rather than replacing it.

Preferred initial architecture:

``` text
postgres
airflow
api
dashboard
```

The AI layer lives inside the API service initially.

Only add a separate RAG service/vector database container if there is a
demonstrated reason.

Ensure:

-   environment variables are documented
-   secrets are not committed
-   health checks exist
-   API starts without optional LLM credentials
-   application remains usable when the LLM provider is unavailable

------------------------------------------------------------------------

# 32. Performance Requirements

The assistant should avoid loading 1M rows into memory.

Reuse the existing server-side cursor / streaming architecture.

Use SQL aggregation whenever possible.

For customer searches:

``` text
SQL filtering
→ LIMIT
→ only retrieve necessary columns
```

Do not:

``` text
Postgres
→ load 1M rows into pandas
→ filter with Python
```

The existing project already solved important memory problems around the
1M-row customer_360 table. Do not regress them.

------------------------------------------------------------------------

# 33. Data/Knowledge Refresh

Build an explicit RAG ingestion command:

``` bash
python -m src.ai.rag.ingest
```

It should:

1.  discover documents
2.  parse
3.  clean
4.  chunk
5.  embed
6.  index
7.  persist metadata
8.  report statistics

Example:

``` text
Documents: 42
Chunks: 1,284
Embedded: 1,284
Indexed: 1,284
Failed: 0
```

Make ingestion idempotent.

Do not duplicate documents when the command runs twice.

------------------------------------------------------------------------

# 34. Evaluation Report

Create:

``` text
report/
├── ai_assistant_evaluation.md
├── retrieval_benchmark.md
├── rag_evaluation.md
├── assistant_failure_analysis.md
└── traces/
```

The reports must contain actual measured results.

Include examples of failures.

A strong project should demonstrate not only:

> "The assistant works."

but also:

> "Here are the cases where it fails, why it fails, and what we
> changed."

------------------------------------------------------------------------

# 35. README Upgrade

Rewrite the README only after implementation is verified.

The README should explain:

## Problem

Businesses have customer data but struggle to convert it into decisions.

## Existing platform

Data pipeline → Customer 360 → ML → recommendations → monitoring.

## New capability

AI assistant that combines:

``` text
Structured analytics
+
ML intelligence
+
Business knowledge
+
LLM reasoning/orchestration
```

## Architecture

Include the final architecture diagram.

## Evaluation

Show real:

-   retrieval metrics
-   answer metrics
-   latency
-   cost
-   citation accuracy
-   failure rate

## Engineering decisions

Explain why:

-   PostgreSQL instead of loading data into pandas
-   hybrid search instead of vector-only
-   reranking
-   controlled SQL
-   evidence validation
-   versioned models
-   caching
-   graceful fallback

## Limitations

Explicitly state that the underlying telecom dataset is synthetic.

Do not make claims about real telecom customer behavior based on this
dataset.

------------------------------------------------------------------------

# 36. What NOT to Claim

Never claim:

-   production customer impact
-   real telecom churn improvement
-   real revenue improvement
-   enterprise deployment
-   real-time data if the data is simulated
-   factual business recommendations not supported by the knowledge
    corpus
-   "hallucination-free"
-   "100% accurate"

Use precise language:

> production-style

> experimentally evaluated

> synthetic-data demonstration

> evidence-grounded

> tested against a benchmark

------------------------------------------------------------------------

# 37. Recommended Implementation Order

Implement in exactly this order unless the existing code requires a
dependency change.

### Stage 0 --- Understand

-   inspect repository
-   run existing tests
-   verify existing services
-   document interfaces

### Stage 1 --- Structured tools

-   customer lookup
-   customer search
-   aggregate analysis
-   churn analysis
-   SHAP
-   recommender
-   retraining analysis

### Stage 2 --- Controlled SQL

-   SQL generation
-   validation
-   timeout
-   row limit
-   read-only execution

### Stage 3 --- RAG

-   document corpus
-   parsing
-   chunking
-   embeddings
-   vector index
-   BM25
-   hybrid retrieval
-   reranking

### Stage 4 --- Agent graph

-   router
-   planner
-   tool execution
-   evidence aggregation
-   insufficient-evidence loop
-   response generation

### Stage 5 --- Guardrails

-   structured output
-   citation validation
-   claim/evidence checks
-   fallback behavior

### Stage 6 --- Evaluation

-   100+ benchmark questions
-   retrieval metrics
-   answer metrics
-   citation metrics
-   latency

### Stage 7 --- Observability

-   traces
-   latency
-   tokens
-   cost
-   errors
-   tool usage

### Stage 8 --- Dashboard

-   assistant UI
-   citations
-   evidence
-   customer-level assistant
-   trace visibility

### Stage 9 --- Deployment

-   Docker
-   health checks
-   environment configuration
-   public demo preparation

### Stage 10 --- Documentation

-   architecture
-   benchmarks
-   failure analysis
-   limitations
-   setup instructions

------------------------------------------------------------------------

# 38. Definition of Done

The project is complete only when all of these are true:

-   [ ] Existing platform still works.
-   [ ] Existing tests still pass.
-   [ ] Assistant can answer structured-data questions.
-   [ ] Assistant can answer customer-specific questions.
-   [ ] Assistant can use churn/SHAP/recommendation outputs.
-   [ ] Assistant can retrieve business documents.
-   [ ] Hybrid BM25 + vector retrieval works.
-   [ ] Reranking works.
-   [ ] Multi-source questions work.
-   [ ] SQL is controlled and read-only.
-   [ ] Answers have evidence.
-   [ ] RAG claims have citations.
-   [ ] Unsupported questions are handled safely.
-   [ ] LLM failure does not destroy core functionality.
-   [ ] 100+ evaluation questions exist.
-   [ ] Retrieval is benchmarked against baselines.
-   [ ] Answer quality is measured.
-   [ ] Citation quality is measured.
-   [ ] Latency is measured.
-   [ ] Token/cost usage is tracked.
-   [ ] Traces are available.
-   [ ] Dashboard exposes the assistant.
-   [ ] Docker setup works.
-   [ ] CI works without external API keys.
-   [ ] README contains real benchmark results.
-   [ ] Synthetic-data limitations remain explicit.

------------------------------------------------------------------------

# 39. Final Instruction to Claude Code

Treat this as an **extension of an existing production-style project**,
not a greenfield rewrite.

Before writing code, inspect the repository and map this plan onto the
existing architecture.

Prioritize:

1.  correctness
2.  reuse
3.  measurable evaluation
4.  explainability
5.  reliability
6.  security
7.  maintainability
8.  performance

Do not implement features merely because they sound impressive.

Every major AI component must have: - a clear purpose - a measurable
evaluation - tests - failure handling - observable behavior

Build incrementally.

After each major stage:

``` bash
pytest
```

and verify the relevant application behavior.

Never replace a verified working implementation with a more complicated
abstraction unless the new implementation provides a measurable benefit.

The final result should feel like a **real AI/data product built on top
of an existing data platform**, not a collection of disconnected AI
demos.
