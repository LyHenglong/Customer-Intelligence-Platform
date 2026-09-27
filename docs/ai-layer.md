# AI layer

Two AI components sit on top of the platform: the **AI Agent Layer** (`src/agents/`), which narrates already-final model output, and the **AI Decision Assistant** (`src/ai/`), which answers questions from tool, SQL and document evidence. Neither changes a prediction.

## AI Agent Layer

Everything above this section is the platform: a calibrated model, a significance-tested recommender, drift monitoring, all producing numbers. `src/agents/` sits entirely on top of that, unchanged - three small agents (Groq, OpenAI-compatible API) that turn already-final model output into plain English for a human. **They explain and draft; they never predict or recommend.** A churn probability, a SHAP attribution, a recommended service - all of that is decided before an agent ever sees it, and no agent's output can change any of it. If Groq is unreachable, every feature that depends on it falls back to showing that same raw data instead of AI prose - nothing breaks, it just gets less readable.

```
SHAP values (already computed,           ──────►  explanation_agent
see explain_churn.py)                             "why is this customer at risk?"
                                                            │
Recommender's top suggested service        ──────►  outreach_agent
(already computed, unchanged)                       "draft a message offering it"
                                                            │
Old vs. new model metrics + PSI drift      ──────►  retrain_summary_agent
(already computed and saved)                        "did this retrain help?"
```

Every example below is real output from this running platform, not illustrative text - see Verified state and Deviations for how each was produced and what went wrong along the way.

### Churn explanation

`GET /explain-churn/{customer_id}` and the dashboard's At-Risk Customers view both call `explain_churn()` with the customer's real churn probability and top-5 SHAP factors (feature, signed value, direction) - the same numbers already shown in the SHAP column, just narrated.

> **Customer `CUST0000269643`, 32.6% churn probability.** SHAP factors: `contract` (+0.544, increases risk), `num_complaints` (+0.380, increases risk), `num_service_calls` (+0.369, increases risk), `late_payments` (+0.286, increases risk), `age` (+0.265, increases risk).
>
> *"The model estimates a 32.6% chance that the customer will churn, driven primarily by the fact that they are on a contract, have logged multiple complaints, and have made several service calls, all of which raise the risk. Additional contributors are late payments and a younger age, which also increase the likelihood of churn."*

One real limitation worth stating plainly: "on a contract" doesn't say *which* contract type. The SHAP feature-name mapping (pre-existing, used identically by the frontend's SHAP column) collapses a one-hot-encoded categorical back to its base column name (`cat__contract_two_year` → `contract`), which loses the specific value. The agent is faithfully narrating what it was given - the imprecision is upstream of it, not invented by it.

### Retention outreach

The frontend's At-Risk Customers view feeds the explanation above, plus the recommender's real top-1 suggestion for the same customer, into `draft_outreach()` (via `POST /outreach-draft/{customer_id}`).

> **Recommended service (from `recommend_for_profile`, unchanged): Internet Service, score 0.948.**
>
> *"We hope you're enjoying your experience with us, and we wanted to reach out personally to ensure everything is running smoothly. We understand you've had a few recent concerns, and we're committed to making things right for you. To help improve your overall experience, we'd like to offer you our enhanced Internet Service, which provides faster speeds and greater reliability. Please let us know if you'd like more details or assistance setting it up.*
>
> *The Retention Team"*

`mentions_service()` (see `src/agents/outreach_agent.py`) confirms the draft actually names the real recommendation rather than a hallucinated one - true here, and logged as a warning (not silently accepted) on the rare case it isn't, verified with a real example in `tests/test_agents.py`.

### Retrain summary

The DAG's `summarize_retrain` task runs after `retrain_churn_model`, comparing the new artifact's metrics against the previous one plus the batch's drift results, and only on the branch where a retrain actually happened - `skip_retrain` produces no new metrics, so there's nothing to summarize on that path (verified: it shows as `skipped` in Airflow on a batch that didn't retrain).

> **Real DAG run, model `20260910T154013Z` vs. previous `20260910T124230Z`:**
>
> *"The new model shows a regression compared to the previous version: the f1 score dropped from 0.2528 to 0.2411 and the ROC‑AUC fell from 0.6693 to 0.6564, both changes exceeding the 0.01 threshold. Precision and recall changed by less than 0.01, so those metrics are essentially unchanged. No significant feature drift was detected in the triggering batch."*

Written to `report/retrain_summaries/retrain_summary_20260910T154013Z.md` and to the `retrain_summaries` Postgres table, and shown on the frontend's Pipeline Status view. The regression itself is expected noise (see [Verified state](engineering-log.md#verified-state) for why: both artifacts trained on the same 150K-row sample with the same random seed, so this reflects sampling variance in the calibration/test split, not a real capability drop) - included here specifically *because* it's the honest case, not the flattering "essentially unchanged" one from an earlier direct test.

### Guardrails

- **Caching**: every explanation/outreach is cached in Postgres, keyed by `(customer_id, agent_type, churn_model_version)` - a retrain invalidates the cache (correctly: the SHAP values it's explaining changed), but a page refresh or re-running the pipeline without a retrain does not. Measured: a cache hit returns in ~0.02s against ~4.5s for a real call.
- **Retry/backoff on rate limits**: `groq_client.complete()` retries up to 3 times with exponential backoff on `RateLimitError`/timeouts. This is not theoretical - generating AI content for 15 real customers in one session genuinely hit Groq's free-tier rate limit mid-run (`429 Too Many Requests`, visible in the container logs), and the backoff recovered every one of them without the feature failing.
- **Graceful fallback everywhere**: every call site (`/explain-churn`, `/outreach-draft/{customer_id}`, the DAG's `summarize_retrain`) catches `AgentCallFailed` and falls back to the raw underlying data (SHAP text, recommendation, metrics dict) rather than raising a 5xx. `GROQ_API_KEY` unset is itself a handled case, not an error - the platform runs completely normally without it, just without the AI prose.
- **Cost bounded on purpose**: the frontend generates AI content only on an explicit button click, one customer at a time (`/outreach-draft/{customer_id}` is a per-customer endpoint, not a batch call) - an enthusiastic user clicking through many rows still fires at most one Groq call per click, and the Postgres cache means re-viewing the same customer costs nothing.

### Enabling it

Optional - everything else in this project works without it. Get a free key at [console.groq.com/keys](https://console.groq.com/keys), add it to `.env`:

```bash
GROQ_API_KEY=gsk_...
```

then restart the `api` container (or `airflow-scheduler` for retrain summaries) so the env var is picked up. `tests/test_agents.py` needs no key at all - every Groq call is mocked.

## AI Decision Assistant security

The assistant can generate SQL, so it is the part of the platform most exposed to prompt injection. It is fenced in three independent layers, so a gap in one does not open the warehouse:

1. **Query validation** (`src/ai/guardrails/sql_safety.py`). One SELECT statement, no comments, no DDL/DML keywords, no system or admin functions (`pg_*`, `dblink`, `current_setting`, `query_to_xml`, ...), and every table it touches must be on the allowlist in `src/ai/config.py`. Tables are found by **parsing the query with sqlglot**, not by regex. A regex over `FROM x`/`JOIN x` used to miss comma joins (`FROM marts.customer_360, public.raw_customers`) and unqualified tables in subqueries (`(SELECT ... FROM pg_user)`); both are now pinned by regression tests.
2. **A read-only transaction with a statement timeout** (`src/ai/tools/sql_tool.py`), and results capped at `AI_SQL_ROW_LIMIT` rows.
3. **A least-privilege database role** (`db/ai_readonly_role.sql`). With `AI_SQL_POSTGRES_USER` set, the tool connects as a role that holds `SELECT` on the allowlisted tables and nothing else, so Postgres itself refuses anything outside them. CI's integration job creates this role and asserts that it cannot read `raw_customers` or `pg_shadow`, and cannot write.

Cost and abuse controls on the HTTP side (`src/api/security.py`):

- The endpoints that can call the LLM (`/assistant/query`, `/explain-churn`, `/outreach-draft`) are **rate limited per client** (`LLM_RATE_LIMIT_PER_MINUTE`, default 10), returning `429` with `Retry-After`.
- Assistant questions are capped at 2,000 characters.
- Optional **API-key auth** (`API_KEYS`) covers every route except `/health`. The Next.js frontend calls the API through its own server-side proxy route (`frontend/app/api/backend/`), which adds the key, so it never reaches the browser.

## Retrieval quality gate

The full benchmark (`python -m src.ai.evaluation.benchmark`, 120 questions) needs Postgres, the embedded corpus and a Groq key, so it runs by hand. The part of retrieval that needs none of those is gated in CI: `src/ai/evaluation/offline_retrieval.py` chunks `knowledge/` in memory and scores BM25 against the benchmark's 20 RAG questions.

| Chunking strategy | Top-1 document accuracy | MRR |
| --- | --- | --- |
| fixed | 1.00 | 1.00 |
| overlapping | 0.90 | 0.93 |
| structure_aware (ingestion default) | 0.75 | 0.85 |

These are BM25 alone, before vector search and the cross-encoder reranker, so they are a floor for the live pipeline rather than a report of it. With six documents, hit@5 is 1.0 by construction, so the gate (`tests/test_ai_evaluation_offline_retrieval.py`) holds top-1 accuracy and MRR just under these values instead. Reproduce with `make eval-retrieval`.
