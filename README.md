# Telecom Customer Churn & Recommendation Platform

[![CI](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-style data platform that predicts telecom customer churn and recommends services to at-risk customers, built on a large-scale synthetic dataset to demonstrate a realistic enterprise data pipeline: simulated batch ingestion, DuckDB processing, a dbt transformation layer with a Customer 360 mart, predictive modeling, and live serving — orchestrated end to end with Airflow.

## Results at a glance

Every number below is measured and reproducible, with the full derivation linked — none of it is an estimate. All figures come from a **synthetic** dataset (see [A note on the data](#a-note-on-the-data)); read the business framing as a demonstration of method, not a real-market claim.

| | |
| --- | --- |
| **Targeting value** | Model-driven targeting nets **$1.41M** on a **$2.26M** offer budget, vs. **$89K** net on a **$6.00M** budget for contacting everyone untargeted — cheaper *and* ~16x more valuable ([Probability calibration](docs/modeling.md#probability-calibration) → `report/findings.md` §6) |
| **Churn model** | LightGBM, **calibrated** (Brier 0.087, beats the always-base-rate baseline of 0.090) — a real probability, not just a ranking score ([Probability calibration](docs/modeling.md#probability-calibration)) |
| **Recommender** | Beats a popularity baseline by **+2.5% MRR**, confirmed significant (bootstrap 95% CI entirely above zero, McNemar p = 2.3×10⁻⁹) — not just eyeballed off a metrics table ([Recommender evaluation](docs/modeling.md#recommender-evaluation)) |
| **Drift monitoring** | PSI-based, verified to fire on injected shifts (18 tests) *and* in a live triggered DAG run — correctly found no drift across 12 real batch comparisons ([Drift monitoring](docs/modeling.md#drift-monitoring)) |
| **Scale & rigor** | 1,000,000 rows end to end; 405 unit tests, a Postgres + dbt integration job and Playwright UI smoke tests green in CI; full Airflow DAG runs verified against a live webserver, not `airflow tasks test` shortcuts ([Engineering quality](#engineering-quality), [Verified state](docs/engineering-log.md#verified-state)) |

## Quick start

```bash
cp .env.example .env                 # set POSTGRES_PASSWORD and AIRFLOW_ADMIN_PASSWORD at minimum
make up                              # Postgres + Airflow (UI on http://localhost:8081)
make serve                           # API (http://localhost:8000/docs) + frontend (http://localhost:3000)
```

The data itself needs a one-time Kaggle download first; see [How to run](#how-to-run). To work on the code without any of that running:

```bash
make install-dev                     # test toolchain + pre-commit hooks
make lint test                       # ruff + 405 unit tests, no database or network needed
make help                            # every other task
```

## The frontend

The platform's user-facing surface is a Next.js app (`frontend/`) that scores all 1,000,000 customers against the latest model artifact. It reads everything through the FastAPI serving layer's dashboard-facing endpoints (`/overview/*`, `/at-risk`, `/customers`, `/pipeline-status`, `/model-history`) rather than Postgres directly, via its own server-side proxy route so the API key never reaches the browser:

- **Overview** — headline KPIs (churn rate, at-risk count/revenue, model AUC) plus top feature importances.
- **At-Risk Customers** — a ranked action list where each row carries a *per-customer* SHAP explanation (▲ = pushes risk up) and a concrete next-best-offer from the recommender, with an opt-in AI Agent Layer section for plain-English risk explanations and drafted retention messages (see [AI layer](docs/ai-layer.md)).
- **Customers** — a searchable customer list and per-customer detail page, a capability the platform's original Streamlit dashboard never had.
- **Segments** — churn rate broken out across demographic/account segments.
- **Model Performance** — the model as it actually is: AUC, recall/precision at the chosen threshold, and top feature importances.
- **Pipeline Status** — the orchestration layer exposed to the same audience: batches ingested, batches remaining until the next conditional retrain, model versions trained to date, and an AI-generated plain-English summary of the most recent real retrain.
- **AI Assistant** — a chat interface over the RAG-backed AI Decision Assistant (see [AI layer](docs/ai-layer.md)).

The platform originally shipped this surface as a Streamlit dashboard; it was retired in favor of this Next.js app.

## Motivation

Customer churn is one of the most common and well-understood problems in data science, and pairing it with a recommendation system demonstrates two distinct modeling techniques within one coherent business narrative: **predict who's about to leave, then recommend something that gives them a reason to stay.**

Rather than analyze a small, pre-cleaned dataset in a notebook, this project is built as a small but complete enterprise-style data platform — using tools and patterns found in real production data teams — to make the underlying engineering, not just the modeling, part of the deliverable.

## A note on the data

This project uses the Kaggle dataset **"Customer Churn Prediction Dataset - 1M"** (`isandeep06/customer-churn-prediction-dataset-1m`), a **large-scale synthetic** telecom churn dataset: exactly 1,000,000 rows, 32 columns, one CSV. It is **not real customer data**, and every pattern or finding described in this project (including `report/findings.md`) reflects the synthetic dataset's generated structure, not an actual telecom market.

Verified properties of the source file (see `src/ingest/split_batches.py` for the profiling used):
- 1,000,000 rows, 1,000,000 unique `customer_id`s (no duplicates)
- Churn label: 90.08% retained (0) / 9.92% churned (1) — imbalanced, as real churn data typically is
- 5 columns have legitimate missing values (not errors): `annual_income` (3.0%), `customer_satisfaction` (1.99%), `num_complaints` (2.99%), `avg_monthly_gb` (5.0%), `credit_score` (4.04%)  
- Row order is already shuffled with respect to churn label and signup date (verified by checking churn rate across 10 sequential slices of the file — all landed within 9.7–10.1%), so splitting the file into sequential batches does not introduce batch-level skew

## Why DuckDB instead of Spark

The natural choice for "big data" processing is often Apache Spark. This project deliberately uses **DuckDB** instead — an in-process analytical database that handles millions of rows efficiently with a fraction of Spark's memory footprint, and has a native dbt adapter. The development machine has **8GB RAM total, and frequently under 1GB free** at any given moment (confirmed during this build - see [Limitations](#limitations)). Running Spark alongside Postgres, Airflow, and everything else simultaneously would not be reliable on hardware like this. DuckDB delivers most of the analytical performance benefit of a distributed engine without the operational overhead, for workloads — like this one — that don't genuinely require a multi-node cluster. In practice, processing a ~77K-row simulated batch (1/13th of the dataset) through DuckDB takes well under a second.

## Architecture

```
Kaggle CSV (1,000,000 rows, synthetic)
        │  split_batches.py (one-time setup)
        ▼
13 simulated "weekly" batch files (data/raw/batch_001.csv ... batch_013.csv)
        │
        ▼
   Airflow DAG (churn_pipeline) — triggered manually per simulated arrival
        │
        ├─ ingest_next_batch ──► DuckDB: type validation (TRY_CAST),
        │                         quality-gate filtering, feature
        │                         engineering (tenure_years,
        │                         total_active_services, avg_gb_per_service)
        │                         ──► writes raw_customers +
        │                             customers_cleaned to Postgres
        │
        ├─ dbt_run  ───────────► staging (5 models) → intermediate (3 models)
        │                         → marts.customer_360 (37 dbt tests)
        ├─ dbt_test
        │
        ├─ detect_feature_drift ► PSI vs the baseline batch across 19
        │                         features ──► public.feature_drift
        │
        └─ check_retrain_needed ─┬─► retrain_churn_model ──► retrain_recommender
                                  │    (every 3rd batch OR any feature
                                  │     PSI ≥ 0.25; both models versioned
                                  │     together, never overwritten)
                                  └─► skip_retrain (otherwise)

customer_360 (one row per customer: demographics, account, services,
              usage, value segmentation, churn label)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  train_churn.py                train_recommender.py
  (LightGBM,                    (k-NN over profile
   class_weight=balanced)        vectors, content-based)
        │                              │
        ▼                              ▼
  models_store/                 models_store/
  churn_model_<ts>.joblib       recommender_<ts>.joblib
        │                              │
        └──────────────┬───────────────┘
                        ▼
              FastAPI (src/api/)
   POST /predict-churn  POST /recommend  GET /explain-churn/{id}
   GET /overview/*  GET /at-risk  GET /customers  GET /pipeline-status
     (loads latest artifact of each at startup, no live DB dependency
      at request time for predictions; dashboard-facing endpoints read
      customer_360 via src/model/dashboard_queries.py; API-key auth,
      LLM rate limiting, /metrics - see Engineering quality)
                        │
                        ├──────────────────────────────┐
                        ▼                              ▼
         AI Agent Layer (src/agents/) — presentation    Next.js frontend (frontend/)
              only, explanation_agent /                  the platform's UI - Overview,
          outreach_agent / retrain_summary_agent          At-Risk Customers, Customers,
           narrate SHAP values, recommendations,           Segments, Model Performance,
          and retrain metrics in plain English via          Pipeline Status, AI Assistant
         Groq - never change a prediction or a
          recommendation, degrade to raw data on
                     any failure
```

## Repository structure

```
.
├── dags/                     # Airflow: churn_pipeline (ingest -> dbt -> drift -> conditional retrain), backup_warehouse
├── src/
│   ├── ingest/               # split the Kaggle CSV into 13 batches; DuckDB clean/validate/engineer, load to Postgres
│   ├── model/                # train/calibrate/evaluate churn model and recommender, SHAP, threshold analysis, registry
│   ├── monitoring/           # PSI drift detection, Slack alerting
│   ├── api/                  # FastAPI app: routers (models, dashboard, assistant), auth + rate limiting, metrics
│   ├── agents/               # AI Agent Layer: narrates SHAP, drafts outreach, summarises retrains (Groq)
│   ├── ai/                   # AI Decision Assistant: router, tools, guarded SQL, hybrid RAG, evaluation, tracing
│   └── warehouse.py          # Postgres connections and server-side-cursor streaming reads
├── dbt/models/               # staging (5) -> intermediate (3) -> marts/customer_360, 37 data tests
├── db/                       # warehouse DDL; least-privilege role for the AI SQL tool
├── frontend/                 # Next.js app: pages, server-side API proxy, Playwright smoke tests
├── knowledge/                # the RAG corpus: retention playbook, policies, service catalog
├── tests/                    # unit tests; tests/integration/ runs against a real Postgres in CI
├── notebooks/                # EDA and statistics, threshold/business-value analysis, model experiments
├── report/                   # business-facing findings, AI-written retrain summaries
├── docs/                     # modeling, AI layer, deployment, engineering log, AI layer build plan
├── docker/                   # Dockerfiles (api, frontend, airflow, mlflow), Caddyfile
├── models_store/             # versioned model metadata; the two served artifacts are committed
├── Makefile                  # `make help` lists the common tasks
└── render.yaml               # Render Blueprint for the public demo
```

## The Customer 360 concept

**Customer 360** is a standard enterprise data pattern: a single, unified table consolidating everything known about a customer — demographics, account details, service subscriptions, usage, and churn risk — instead of that information being scattered across tables that must be joined every time it's needed.

`marts.customer_360` is dbt's final mart model: one row per customer, built by joining 5 staging models and 3 intermediate models. It is the **only** table the churn model, the recommender, and the serving layer read from.

### dbt lineage

```
sources: public.customers_cleaned, public.raw_customers  (written by batch_loader.py)
        │
        ▼
staging/  stg_customers__demographics   stg_customers__account   stg_customers__services
          stg_customers__usage          stg_customers__churn
        │
        ▼
intermediate/  int_customer_service_summary   (services x account: cost per active service)
               int_customer_usage_summary     (usage x account: complaints per tenure-month, disengagement flag)
               int_customer_value_segment     (demographics x account: income/spend ratio, tenure bucket)
        │
        ▼
marts/  customer_360   (all of the above, joined on customer_id)
```

37 dbt tests (`not_null`, `unique`, `accepted_values`) run across all three layers — all passing as of the last verified run (see [Verified state](docs/engineering-log.md#verified-state)).

## Modeling highlights

Full detail, with how each number was produced: [docs/modeling.md](docs/modeling.md).

- **Statistics before modeling.** Chi-square and Mann-Whitney tests with effect sizes, survival analysis (Kaplan-Meier, Cox PH) and clustering in `notebooks/eda_and_statistical_analysis.ipynb`. Contract type and tenure carry the signal; no demographic "churn persona" is supported by the data.
- **A calibrated churn model.** LightGBM with `class_weight="balanced"` ranks well but overstated churn about 4x. Platt scaling on a held-out calibration split brings the Brier score from 0.196 to 0.087, beating the 0.090 always-base-rate baseline, with AUC unchanged.
- **Decision thresholds from business value.** The threshold is re-derived at every retrain as the highest cut-off still meeting a recall floor, and checked against an expected-value model of offer cost against retained revenue, not left at 0.5.
- **A recommender tested against baselines.** Leave-one-out evaluation on customers outside the k-NN reference set: +2.5% MRR over a popularity baseline, confirmed by a paired bootstrap and McNemar's test.
- **Drift monitoring that is tested to fire.** PSI on 19 features for every batch, with reference-only bin edges, NULLs as their own bucket and open outer bins. It correctly finds no drift in this pre-shuffled data; 18 tests inject real shifts and confirm it catches them.
- **An honest ceiling.** The synthetic data caps AUC around 0.67. The identical training code reaches 0.83 on the real IBM Telco dataset, so the limit is the data, not the method.

## AI layer

Details, real examples and the security model: [docs/ai-layer.md](docs/ai-layer.md).

- **AI Agent Layer** (`src/agents/`): explains a customer's SHAP factors in plain English, drafts a retention message for the recommended service, and summarises each retrain. It narrates final model output and never changes a prediction; every call falls back to the raw data if Groq is unavailable, and results are cached per model version.
- **AI Decision Assistant** (`src/ai/`): routes a question to structured tools, guarded SQL or hybrid RAG (BM25 + pgvector, fused and reranked), then answers only from the evidence those return, with citations and a stored trace.
- **Fenced SQL.** Generated SQL is parsed with sqlglot and checked against a table allowlist, runs read-only with a timeout, and can run as a Postgres role that has SELECT on the allowlisted tables only.

## Engineering quality

| | |
| --- | --- |
| **CI** | Every push runs ruff, 405 unit tests with a 75% coverage floor, an integration job that builds the warehouse in a real pgvector Postgres (schema, ingestion, `dbt build` with all data tests, the AI SQL role), and the frontend's eslint, typecheck, production build and Playwright smoke tests. |
| **CD** | After CI passes on `main`, API and frontend images go to GitHub Container Registry, tagged `latest` and with the commit SHA. |
| **API security** | Optional API-key auth; a server-side proxy keeps the key out of the browser; per-client rate limits on the LLM-backed endpoints; bounded inputs. |
| **SQL safety** | Parser-based validation, read-only transactions and a least-privilege role; the role's refusals are asserted in CI. |
| **Observability** | Request IDs, structured (JSON) access logs and Prometheus metrics at `/metrics`, plus per-request AI traces with tokens and cost. |
| **Retrieval quality** | An offline gate scores BM25 over the real `knowledge/` corpus against the benchmark's RAG questions on every CI run. |
| **Tooling** | `pyproject.toml` (ruff, pytest, coverage), pre-commit hooks, a Makefile and grouped Dependabot updates. See [CONTRIBUTING.md](CONTRIBUTING.md). |

## How to run

### 1. Prerequisites

- Docker Desktop
- A `kaggle.json` API token at `~/.kaggle/kaggle.json` (get one from kaggle.com → Account → Create New API Token) — only needed once, to download the dataset
- Copy `.env.example` to `.env` and fill in real values (a Postgres password and Airflow admin password at minimum)
- Optional: a free `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys) for the [AI layer](docs/ai-layer.md) — everything else works without it

### 2. One-time data setup

```bash
kaggle datasets download -d isandeep06/customer-churn-prediction-dataset-1m -p data/raw_download --unzip
python src/ingest/split_batches.py   # writes data/raw/batch_001.csv ... batch_013.csv
```

### 3. Start the warehouse + orchestration

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

Airflow UI: http://localhost:8081 (login from `.env`'s `AIRFLOW_ADMIN_USER`/`AIRFLOW_ADMIN_PASSWORD`; 8081, not the default 8080 — see [Deviations](docs/engineering-log.md#deviations-from-the-original-spec-and-why)). Trigger the `churn_pipeline` DAG manually — each run ingests one more batch (simulating one weekly arrival), then runs dbt, then conditionally retrains.

### 4. Train models manually (or let the DAG's retrain step do it)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; use bin/ on macOS/Linux
python -m src.model.train_churn
python -m src.model.train_recommender
```

### 5. Serve

```bash
docker compose --profile serving up -d api frontend
```

- API docs: http://localhost:8000/docs
- Frontend (Next.js): http://localhost:3000

`api` and `frontend` are behind Docker Compose's `serving` profile, so a plain `docker compose up` (warehouse + orchestration only) doesn't also pay for containers you may not be actively using — deliberate, given the RAM constraint. The Next.js app under [`frontend/`](frontend/) is the platform's UI. It reaches `api` through its own server-side proxy route, so setting `API_KEYS` on the API (and the matching `API_KEY` for the frontend) locks the API down without exposing the key.

To run the frontend outside Docker for local development (hot reload):

```bash
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

### 6. Deploy a public demo (optional)

The API and frontend deploy to Render against a Neon Postgres using [`render.yaml`](render.yaml). The step-by-step walkthrough, and the production hardening around it (TLS, secrets, auth, scaling, backups, alerting, observability), is in [docs/deployment.md](docs/deployment.md).

### dbt directly

```bash
cd dbt
export DBT_PROFILES_DIR=$(pwd)   # profiles.yml lives in the project, not ~/.dbt/
dbt build                        # models + all 37 data tests
```

### Tests

```bash
make test                # 405 unit tests: no database, Docker, model training or Groq key needed
make coverage            # the same, with a coverage report (CI enforces 75%)
make eval-retrieval      # offline retrieval quality over knowledge/
make frontend-check frontend-test   # eslint, typecheck, build, then Playwright smoke tests
```

The integration tests (`tests/integration/`) need a warehouse built the way CI's `integration` job builds one, and run with `make test-integration`.

## Tech stack

| Category | Tools |
|---|---|
| Languages | Python, SQL, TypeScript |
| Orchestration | Apache Airflow 2.10.3 (LocalExecutor) |
| Processing | DuckDB |
| Transformation | dbt-core 1.8 / dbt-postgres |
| Storage | PostgreSQL 16 + pgvector |
| Modeling | LightGBM, scikit-learn (k-NN, K-means, calibration), pandas |
| Statistical analysis | scipy (chi-square, Mann-Whitney U), statsmodels (VIF), lifelines (Kaplan-Meier, Cox PH) |
| Explainability | SHAP (TreeExplainer) |
| Model registry | MLflow |
| AI | Groq (`openai/gpt-oss-120b` / `-20b`), sentence-transformers (embeddings, cross-encoder reranking), rank-bm25, sqlglot |
| Serving | FastAPI, Prometheus metrics |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, Recharts |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions, GitHub Container Registry, Render |
| Quality | pytest, Playwright, dbt tests, ruff, pre-commit |

## Limitations

- **The dataset is synthetic, not real telecom customer data.** Every "finding" in `report/findings.md` and the dashboard describes patterns the data generator produced, not an actual market.
- **DuckDB was used in place of Spark due to local hardware constraints (8GB RAM, often <1GB free in practice).** The pipeline's per-batch DuckDB step is fast at this scale (~77K rows/batch, well under a second); it was not tested against a distributed engine at larger scale.
- **Simulated batch ingestion replays a static, pre-existing dataset rather than genuine streaming arrivals**, and batches are triggered manually rather than on a wall-clock schedule (see [Deviations](docs/engineering-log.md#deviations-from-the-original-spec-and-why)).
- **The recommendation system uses content-based filtering only** (k-NN over demographic/account/usage profile vectors, recommending services popular among a customer's nearest neighbors). A real production system would likely combine this with collaborative filtering and online feedback signals.
- **Drift-triggered retraining is implemented, but this dataset cannot exercise it.** The pipeline computes per-feature PSI for every arriving batch and retrains when any feature crosses the 0.25 band (see [Drift monitoring](docs/modeling.md#drift-monitoring)). Because all 13 batches are slices of one pre-shuffled file, the measured drift is ~0.0004 across every feature and the cadence rule (every 3rd batch) is what actually fires in practice. The detector's ability to fire is demonstrated in `tests/test_drift.py`, not by this data.
- **Churn model performance is modest, and this was investigated, not assumed.** A dedicated experiment (`notebooks/model_dev_offline.py`) compared 4 model families, SMOTE vs. class-weighting, and cross-validated the winner on the full 1M-row dataset. Best result: LightGBM, AUC 0.683 ± 0.002 (5-fold CV) — a small, real improvement over the original RandomForest (0.677), but nothing tried closes the gap to a "strong" classifier. This looks like a genuine ceiling in the synthetic data's individual-level signal (segment-level signal is much stronger — e.g. contract type alone separates a 4.8x churn-rate gap) rather than a fixable modeling gap.
- **`/recommend` answers for any customer in the warehouse, but not identically fast for all of them.** The recommender's k-NN reference set is deliberately bounded (~154K profiles) to keep the artifact small; a customer outside it (about 85% of the 1M, so the normal case) triggers one indexed warehouse lookup and is then matched against that index. Coverage is 100% — verified on a random sample of 40 real customer IDs — but those requests do touch Postgres, unlike `/predict-churn`, which remains purely in-memory.
- **Per-customer explanations are real SHAP values, computed only for the displayed subset.** `TreeExplainer` runs against the bounded set of rows actually on screen (at most 500), never the full 1M-row table — an intentional cost/scale trade-off, not a full-population attribution.
- **The AI Agent Layer is a presentation layer over frozen predictions, not a modeling improvement** - it never sees a feature the model didn't already use, and it cannot change a churn probability, a SHAP ranking, or a recommendation. Two real, specific caveats from live verification: (1) SHAP factor names for categorical features collapse to the base column (e.g. "contract", not "month-to-month") because that mapping is pre-existing and shared with the dashboard's SHAP column, so an explanation can say a factor matters without saying which value of it does; (2) the exact Groq model IDs pinned here (`openai/gpt-oss-120b`/`-20b`) are a live external dependency that already moved once during this project (see [Deviations](docs/engineering-log.md#deviations-from-the-original-spec-and-why)) and could again - unlike a pinned pip package, there's no local copy to fall back to, only the graceful-degradation-to-raw-data behavior this layer was built with from the start.
- **The LLM rate limiter counts per process.** It needs no Redis, which suits a single free-tier instance, but with several API replicas each enforces its own limit, so the real ceiling is replicas times the configured rate. A shared store would be the fix for a horizontally scaled deployment.

## Documentation

- [docs/modeling.md](docs/modeling.md): statistical analysis, drift monitoring, calibration, recommender evaluation
- [docs/ai-layer.md](docs/ai-layer.md): AI Agent Layer, AI Decision Assistant, SQL and API security, retrieval quality
- [docs/deployment.md](docs/deployment.md): public demo on Render + Neon, production hardening
- [docs/engineering-log.md](docs/engineering-log.md): what was verified and how, and every deviation from the spec
- [report/findings.md](report/findings.md): business-facing findings
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup and conventions

## License

[MIT](LICENSE) — the code is free to use, modify, and learn from. The dataset it runs on is a third-party synthetic Kaggle dataset (`isandeep06/customer-churn-prediction-dataset-1m`, see [A note on the data](#a-note-on-the-data)) and is not redistributed here; it is not covered by this license.

## Author

Ly Henglong (Penguin) — Final-year ICT student, American University of Phnom Penh, focused on data science and machine learning.

- LinkedIn: [linkedin.com/in/lyhenglong](https://linkedin.com/in/lyhenglong)
- GitHub: [github.com/LyHenglong](https://github.com/LyHenglong)
