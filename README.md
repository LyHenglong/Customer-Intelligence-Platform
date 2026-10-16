# Telecom Customer Churn & Recommendation Platform

[![CI](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-style data platform that predicts telecom customer churn and recommends services to at-risk customers — built on a large-scale synthetic dataset to demonstrate a realistic enterprise pipeline: simulated batch ingestion, DuckDB processing, a dbt transformation layer, predictive modeling, and live serving, orchestrated end to end with Airflow.

## Results at a glance

Every number below is measured and reproducible, with the full derivation linked — none of it is an estimate. All figures come from a **synthetic** dataset (see [A note on the data](#a-note-on-the-data)); read the business framing as a demonstration of method, not a real-market claim.

| | |
| --- | --- |
| **Targeting value** | Model-driven targeting nets **$1.41M** on a **$2.26M** offer budget, vs. **$89K** net on a **$6.00M** budget for contacting everyone untargeted — cheaper *and* ~16x more valuable ([Probability calibration](#probability-calibration) → `report/findings.md` §6) |
| **Churn model** | LightGBM, **calibrated** (Brier 0.087, beats the always-base-rate baseline of 0.090) — a real probability, not just a ranking score |
| **Recommender** | Beats a popularity baseline by **+2.5% MRR**, confirmed significant (bootstrap 95% CI entirely above zero, McNemar p = 2.3×10⁻⁹) — not just eyeballed off a metrics table |
| **Drift monitoring** | PSI-based, verified to fire on injected shifts (18 tests) *and* in a live triggered DAG run — correctly found no drift across 12 real batch comparisons |
| **Scale & rigor** | 1,000,000 rows end to end, full test suite green in CI, full Airflow DAG runs verified against a live webserver (not shortcut-tested) ([engineering log](docs/ENGINEERING_LOG.md)) |

## What it does

- **Predicts churn** per customer with a calibrated LightGBM model, explained per-row with real SHAP values.
- **Recommends a retention offer** — a content-based k-NN recommender, scoped to services the customer doesn't already have.
- **Monitors data drift** on every incoming batch (PSI) and conditionally retrains both models when the data actually shifts, not just on a fixed schedule.
- **Serves it all** through a FastAPI backend and a Next.js frontend: an Overview page, a ranked At-Risk list, customer search, segment breakdowns, model performance over time, pipeline status, and an AI chat assistant.
- **Narrates results in plain English**, optionally — an AI Agent Layer (Groq) turns SHAP values, recommendations, and retrain metrics into readable text. It never predicts or recommends anything itself, and everything works identically without it.

The platform originally shipped this surface as a Streamlit dashboard; it was retired in favor of the Next.js app (see git history for the Streamlit-era screenshots).

## A note on the data

This project uses the Kaggle dataset **"Customer Churn Prediction Dataset - 1M"** (`isandeep06/customer-churn-prediction-dataset-1m`) — a **large-scale synthetic** telecom churn dataset: 1,000,000 rows, 32 columns, one CSV. It is **not real customer data**, and every finding in this project reflects the synthetic generator's structure, not an actual market. Verified properties: 1,000,000 unique customer IDs (no duplicates), 90.08%/9.92% retained/churned, 5 columns with legitimate (non-error) missing values, and row order already shuffled with respect to churn and signup date, so sequential batching introduces no batch-level skew.

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
        ├─ ingest_next_batch ──► DuckDB: type validation, quality-gate
        │                         filtering, feature engineering
        │                         ──► writes to Postgres
        │
        ├─ dbt_run / dbt_test ─► staging (5 models) → intermediate (3) →
        │                         marts.customer_360 (37 dbt tests)
        │
        ├─ detect_feature_drift ► PSI vs. the baseline batch, 19 features
        │                         ──► public.feature_drift
        │
        └─ check_retrain_needed ─┬─► retrain_churn_model + retrain_recommender
                                  │    (every 3rd batch OR any feature PSI ≥ 0.25)
                                  └─► skip_retrain (otherwise)

customer_360 ──┬──► train_churn.py (LightGBM) ──► models_store/churn_model_<ts>.joblib
               └──► train_recommender.py (k-NN) ─► models_store/recommender_<ts>.joblib
                              │
                              ▼
                    FastAPI (src/model/api.py)
        predict-churn · recommend · explain-churn · overview/* · at-risk · customers
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    AI Agent Layer (src/agents/)      Next.js frontend (frontend/)
    presentation only — narrates     the platform's UI: Overview,
    SHAP/recommendations/retrain      At-Risk, Customers, Segments,
    metrics, never changes them      Model Performance, Pipeline Status, AI Assistant
```

<details>
<summary><strong>Full repository structure</strong></summary>

```
.
├── dags/churn_pipeline.py          # Airflow DAG: ingest -> dbt run/test -> drift check -> conditional retrain
├── src/
│   ├── warehouse.py                 # server-side-cursor streaming reads (large tables never load client-side)
│   ├── ingest/                      # split_batches.py (one-time), batch_loader.py (DuckDB clean/validate/load)
│   ├── model/
│   │   ├── train_churn.py           # trains, calibrates, evaluates the churn classifier
│   │   ├── train_recommender.py     # trains the content-based recommender
│   │   ├── evaluate_recommender.py  # leave-one-out offline evaluation vs. baselines
│   │   ├── explain_churn.py         # shared per-customer SHAP logic
│   │   ├── dashboard_queries.py     # warehouse/filesystem reads behind api.py's dashboard endpoints
│   │   └── api.py                   # FastAPI app: predictions, dashboard REST endpoints, AI assistant
│   ├── monitoring/drift.py          # PSI drift detection between batches
│   └── agents/                      # AI Agent Layer - presentation only
├── frontend/                        # Next.js UI, talks to api.py's REST endpoints
├── dbt/models/{staging,intermediate,marts}/
├── db/schema.sql                    # warehouse DDL
├── notebooks/                       # offline model comparison, hyperparameter sweep, EDA, threshold analysis
├── tests/                           # ingestion, model math, drift, recommender eval, dashboard queries, agents
├── report/findings.md               # business-facing write-up
├── docs/ENGINEERING_LOG.md          # full build log: every verification run, bug, and deviation
├── docker/                          # Dockerfile.api / .frontend / .airflow / .mlflow, Caddyfile for self-hosted TLS
├── scripts/                         # one-off deployment/migration/precompute scripts
├── docker-compose.yml
├── render.yaml                      # Render Blueprint for the public demo deployment
└── .github/workflows/ci.yml
```

</details>

## Tech stack

| Category | Tools |
|---|---|
| Languages | Python, SQL |
| Orchestration | Apache Airflow 2.10.3 (LocalExecutor) |
| Processing | DuckDB |
| Transformation | dbt-core 1.8 / dbt-postgres |
| Storage | PostgreSQL 16 |
| Modeling | LightGBM, scikit-learn (k-NN, K-means, preprocessing), pandas |
| Statistical analysis | scipy (chi-square, Mann-Whitney U), statsmodels (VIF), lifelines (Kaplan-Meier, Cox PH) |
| Explainability | SHAP (TreeExplainer) |
| AI Agent Layer | Groq (`openai/gpt-oss-120b` / `-20b`), presentation only — see [AI Agent Layer](#ai-agent-layer) |
| Model serving | FastAPI |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, Recharts |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest, dbt tests |

## The Customer 360 concept

`marts.customer_360` is a standard enterprise data pattern: one unified row per customer — demographics, account, services, usage, churn risk — built by dbt from 5 staging models and 3 intermediate models. It is the **only** table the churn model, the recommender, and the serving layer read from.

```
sources: raw_customers, customers_cleaned
   → staging (demographics, account, services, usage, churn)
   → intermediate (service summary, usage summary, value segment)
   → marts.customer_360
```

37 dbt tests (`not_null`, `unique`, `accepted_values`) run across all three layers.

## Statistical & analytical depth

`notebooks/eda_and_statistical_analysis.ipynb` is a fully executed analysis notebook that goes beyond descriptive charts into real statistical methodology:

- **Significance testing** — chi-square (Cramer's V) for categorical features, Mann-Whitney U for numeric ones. `contract` and `tenure_bucket` are genuinely associated with churn; demographic features (`gender`, `education`, `marital_status`, `payment_method`) are not.
- **Statistical vs. practical significance** — called out explicitly where a large sample makes p < 0.05 trivial (e.g. `monthlycharges`, hazard ratio ≈ 0.998/dollar).
- **Survival analysis** — Kaplan-Meier curves and a Cox Proportional Hazards model, modeling *time to* churn rather than discarding it into a binary label. `is_month_to_month` hazard ratio ≈ 2.77; concordance index 0.617.
- **Unsupervised segmentation** — K-means with elbow/silhouette selection and a PCA projection. Reported honestly: silhouette ≈ 0.14–0.16, a soft cluster structure, not oversold as sharply separated.
- **SHAP explainability** — the same per-customer SHAP logic also powers the frontend's live At-Risk Customers view.

```bash
pip install -r requirements.txt -r requirements-notebooks.txt   # separate: not installed into the API image or CI
jupyter lab notebooks/
```

See `report/findings.md` for the business-facing summary of these findings.

## Drift monitoring

Every incoming batch is scored against a fixed baseline using the **Population Stability Index** — `PSI = Σ (actual% - expected%) × ln(actual% / expected%)` — across 19 features. Retraining fires on two triggers: a fixed cadence (every 3rd batch, so the model never silently goes stale) and an evidence trigger (any feature PSI ≥ 0.25).

Three implementation details that matter: bin edges come from the reference distribution only (never re-binned per batch, or every batch looks identical to itself); NULLs are their own bucket rather than dropped (a missing-rate jump is itself a drift signal); and outer bin edges are open (`±inf`) so out-of-range values land in the end bins instead of being silently discarded.

**The honest result on this dataset: no drift**, as expected — all 13 batches are slices of one pre-shuffled file, so the maximum PSI observed across every feature and batch pair is 0.0006. That the detector *can* fire is demonstrated separately in `tests/test_drift.py` (18 tests), which injects real shifts (mean/variance changes, missingness jumps, unseen categories, out-of-range values) and asserts they're caught.

## Probability calibration

`class_weight="balanced"` makes the model's *ranking* work on a ~10%-imbalanced target, but inflates its raw scores — mean predicted probability was 0.41 against an actual churn rate of 0.10. `train_churn.py` fixes this with a three-way split (train/calibration/test) and `CalibratedClassifierCV` (sigmoid), fit only on the held-out calibration split:

| | Brier score | Mean predicted | Actual rate |
| --- | --- | --- | --- |
| Uncalibrated | 0.196 (worse than the 0.090 baseline) | 0.408 | 0.100 |
| **Calibrated** | **0.087** (beats the baseline) | **0.100** | 0.100 |

Calibration changes what the numbers *mean*, not what the model *does* — sigmoid scaling is monotonic, so ranking, AUC (0.6693), precision, and recall are unchanged. What changes is the decision threshold, which moves from 0.451 to 0.105 to match where a real probability near the base rate actually sits. Full before/after, including the expected-value threshold analysis, is in `report/findings.md` §6.

## Recommender evaluation

Unlike the other models here, the recommender's metadata used to record only structural facts. `src/model/evaluate_recommender.py` closes that gap with a leave-one-out protocol — hide one service a customer genuinely owns, present them as if they didn't, and check where it lands in the ranking — scored on customers held entirely outside the k-NN reference set:

| Ranker | Hit@1 | Hit@3 | MRR |
| --- | --- | --- | --- |
| **k-NN (production)** | **0.519** | 0.878 | **0.703** |
| Popularity baseline | 0.483 | 0.891 | 0.686 |
| Random | 0.245 | 0.681 | 0.498 |

**The honest result: k-NN beats popularity, but by a small margin** against a genuinely hard baseline (8 services with skewed adoption rates leave little room to be bad). Significance tested two ways rather than reported and left alone: a paired bootstrap puts the MRR gap at +0.0170 (95% CI [+0.0105, +0.0237]), and an exact McNemar test on Hit@1 gives p = 2.3×10⁻⁹ — both agree the edge is real, reported as small, not rounded up. Full detail in `report/findings.md` §5a.

## AI Agent Layer

Three small agents (Groq) that turn already-final model output into plain English. **They explain and draft; they never predict or recommend** — a churn probability, a SHAP attribution, a recommended service are all decided before an agent sees them, and nothing an agent produces can change any of it. If Groq is unreachable, every dependent feature falls back to showing the same raw data instead of AI prose.

> **Customer `CUST0000269643`, 32.6% churn probability.** *"The model estimates a 32.6% chance that the customer will churn, driven primarily by the fact that they are on a contract, have logged multiple complaints, and have made several service calls, all of which raise the risk..."*

Real output from this running platform (via `GET /explain-churn/{id}`), not illustrative text — see the [engineering log](docs/ENGINEERING_LOG.md) for the retention-outreach and retrain-summary examples, and for exactly how each was produced.

**Guardrails**: every explanation/draft is cached in Postgres keyed by `(customer_id, agent_type, model_version)` — a hit returns in ~0.02s against ~4.5s for a real call, and a retrain correctly invalidates it. `groq_client.complete()` retries with exponential backoff on rate limits (verified against a real `429` mid-session, not just mocked). Every call site catches failures and falls back to the raw data rather than raising a 5xx — `GROQ_API_KEY` unset is a handled case, not an error. AI content generates on an explicit per-customer click, never in batch, so cost stays bounded.

Optional — get a free key at [console.groq.com/keys](https://console.groq.com/keys), add `GROQ_API_KEY=gsk_...` to `.env`, restart the `api` container. Tests need no key at all; every Groq call is mocked.

## How to run

### 1. Prerequisites

- Docker Desktop
- A `kaggle.json` API token at `~/.kaggle/kaggle.json` (kaggle.com → Account → Create New API Token) — only needed once
- Copy `.env.example` to `.env` and fill in real values (a Postgres password and Airflow admin password at minimum)
- Optional: a free `GROQ_API_KEY` for the [AI Agent Layer](#ai-agent-layer) — everything else works without it

### 2. One-time data setup

```bash
kaggle datasets download -d isandeep06/customer-churn-prediction-dataset-1m -p data/raw_download --unzip
python src/ingest/split_batches.py   # writes data/raw/batch_001.csv ... batch_013.csv
```

### 3. Start the warehouse + orchestration

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

Airflow UI: http://localhost:8081. Trigger the `churn_pipeline` DAG manually — each run ingests one more simulated batch, then runs dbt, then conditionally retrains.

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

API docs: http://localhost:8000/docs · Frontend: http://localhost:3000

To run the frontend outside Docker for local development (hot reload):

```bash
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

### 6. Public demo deployment (optional)

Both the API and the frontend can run reachable from anywhere — [Neon](https://neon.tech) (free Postgres) plus [Render](https://render.com) (free Docker-based web services) — without cloning the repo or running Docker locally. Scoped to just these two services: Airflow stays local-only, and the warehouse the API reads is a read-mostly copy, not the live pipeline.

1. Create a free [Neon](https://neon.tech) project.
2. Migrate the data the API actually reads: `docker compose up -d postgres && ./scripts/migrate_to_neon.sh "postgresql://user:pass@host/dbname?sslmode=require"`.
3. Push this repo to GitHub (already done).
4. On [Render](https://render.com): **New → Blueprint**, point it at this repo. [`render.yaml`](render.yaml) creates both `telecom-churn-api` and `telecom-churn-frontend` automatically.
5. Fill in `telecom-churn-api`'s Postgres env vars with the Neon project from step 1. `GROQ_API_KEY` is optional. Leave `CORS_ALLOWED_ORIGINS`/`NEXT_PUBLIC_API_URL` as placeholders — neither service's real URL exists yet.
6. Once both are deployed, copy their real URLs.
7. Set `CORS_ALLOWED_ORIGINS` (on the API) and `NEXT_PUBLIC_API_URL` (on the frontend) to those URLs, then **redeploy the frontend specifically** — that value is baked into the browser bundle at build time, so a plain env var update alone won't take effect.
8. Precompute churn scores against the same Neon database, so the free-tier API never has to score all 1,000,000 customers in-process: `python -m scripts.precompute_churn_scores` (with `POSTGRES_*` pointed at Neon). `render.yaml` already sets `USE_PRECOMPUTED_SCORES=true`, which makes the dashboard endpoints read this table with a plain `SELECT` instead. Re-run after every retrain that promotes a new model.

Two things worth knowing: Render's free tier spins down after ~15 minutes idle (first request after that pays a 10-50s cold start — expected, not a bug), and this deployment always serves whatever was newest in `models_store/` at the last image build (no MLflow service alongside it).

### Tests

```bash
pytest tests/ -v
```

Fully mocked — no database, no Docker, no trained model artifact, no real Groq key required, which is what lets CI run on a clean checkout. Covers ingestion logic, expected-value/threshold math, recommender invariants, PSI drift, recommender evaluation metrics, dashboard query logic, and the AI Agent Layer's prompt grounding and failure handling.

## Production hardening

Real, working configuration — none of it requires a paid service, and each piece degrades gracefully to "off" if its optional env var is unset.

- **TLS**: automatic on Render (terminated at their edge); [`docker/Caddyfile`](docker/Caddyfile) covers self-hosting with a real domain instead (Let's Encrypt, automatic renewal).
- **Secrets**: no dedicated secrets manager — every credential comes from the environment (Render's env var UI, or a gitignored local `.env`), matching this project's "avoid unnecessary infrastructure" pattern elsewhere.
- **CI/CD**: [`ci.yml`](.github/workflows/ci.yml) runs the test suite on every push/PR; [`cd.yml`](.github/workflows/cd.yml) builds and pushes versioned Docker images to GHCR once CI passes on `main`.
- **Horizontal scaling**: `api.py` is stateless at request time (verified — nothing writes to local disk, everything persists to Postgres), so `render.yaml`'s commented-out `numInstances` is a one-line change away on a paid plan.
- **Backups**: [`backup_warehouse.py`](dags/backup_warehouse.py) runs daily, gzips to `./backups/`, prunes after 14 days — skips itself automatically against a hosted provider like Neon, which already backs up natively.
- **Alerting**: [`alerting.py`](src/monitoring/alerting.py) posts to a Slack webhook on significant drift or a failed-and-exhausted-retries pipeline task; silently a no-op if unset.

## Limitations

- The dataset is **synthetic**, not real telecom data — every "finding" describes what the generator produced, not an actual market.
- Churn model performance is modest (AUC ≈ 0.68) and investigated, not assumed — see the real-data signal benchmark in the [engineering log](docs/ENGINEERING_LOG.md). The ceiling is in the data (27 of 38 features are individually indistinguishable from noise), not the modeling: the identical method on real IBM Telco data reaches 0.83.
- The recommender is content-based only (k-NN over profile vectors); a real production system would likely add collaborative filtering and online feedback.
- Drift-triggered retraining is implemented and tested (`tests/test_drift.py`), but this particular dataset — pre-shuffled slices of one file — cannot exercise it for real; the fixed-cadence trigger is what actually fires in practice here.
- Per-customer SHAP explanations are computed only for the rows actually displayed (at most 500), not the full population — an intentional cost/scale trade-off.
- The AI Agent Layer narrates frozen predictions; it cannot see a feature the model didn't use or change a ranking. SHAP factor names for categorical features collapse to the base column name (e.g. "contract", not "month-to-month").
- DuckDB was used in place of Spark due to local hardware constraints (8GB RAM) — fast at this scale, untested against a distributed engine at larger scale.

See [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md) for the full verification record and every deviation from the original design, with the reasoning behind each one, and [`report/findings.md`](report/findings.md) for the business-facing write-up.

## License

[MIT](LICENSE) — the code is free to use, modify, and learn from. The dataset it runs on is a third-party synthetic Kaggle dataset (see [A note on the data](#a-note-on-the-data)) and is not redistributed here; it is not covered by this license.

## Author

Ly Henglong — Final-year ICT student, American University of Phnom Penh, focused on data science and machine learning.

- LinkedIn: [linkedin.com/in/lyhenglong](https://linkedin.com/in/lyhenglong)
- GitHub: [github.com/LyHenglong](https://github.com/LyHenglong)
