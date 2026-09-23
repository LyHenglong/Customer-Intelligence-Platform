# Telecom Customer Churn & Recommendation Platform

[![CI](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/LyHenglong/Customer-Intelligence-Platform/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-style data platform that predicts telecom customer churn and recommends services to at-risk customers, built on a large-scale synthetic dataset to demonstrate a realistic enterprise data pipeline: simulated batch ingestion, DuckDB processing, a dbt transformation layer with a Customer 360 mart, predictive modeling, and live serving — orchestrated end to end with Airflow.

## Results at a glance

Every number below is measured and reproducible, with the full derivation linked — none of it is an estimate. All figures come from a **synthetic** dataset (see [A note on the data](#a-note-on-the-data)); read the business framing as a demonstration of method, not a real-market claim.

| | |
| --- | --- |
| **Targeting value** | Model-driven targeting nets **$1.41M** on a **$2.26M** offer budget, vs. **$89K** net on a **$6.00M** budget for contacting everyone untargeted — cheaper *and* ~16x more valuable ([Probability calibration](#probability-calibration) → `report/findings.md` §6) |
| **Churn model** | LightGBM, **calibrated** (Brier 0.087, beats the always-base-rate baseline of 0.090) — a real probability, not just a ranking score ([Probability calibration](#probability-calibration)) |
| **Recommender** | Beats a popularity baseline by **+2.5% MRR**, confirmed significant (bootstrap 95% CI entirely above zero, McNemar p = 2.3×10⁻⁹) — not just eyeballed off a metrics table ([Recommender evaluation](#recommender-evaluation)) |
| **Drift monitoring** | PSI-based, verified to fire on injected shifts (18 tests) *and* in a live triggered DAG run — correctly found no drift across 12 real batch comparisons ([Drift monitoring](#drift-monitoring)) |
| **Scale & rigor** | 1,000,000 rows end to end, 63/63 tests green in CI, full Airflow DAG runs verified against a live webserver (not `airflow tasks test` shortcuts) ([Verified state](#verified-state)) |

## The frontend

The platform's user-facing surface is a Next.js app (`frontend/`) that scores all 1,000,000 customers live against the latest model artifact via the FastAPI serving layer's dashboard-facing REST endpoints (`/overview/*`, `/at-risk`, `/customers`, `/pipeline-status`, `/model-history`) rather than reading Postgres directly:

- **Overview** — headline KPIs (churn rate, at-risk count/revenue, model AUC) plus top feature importances.
- **At-Risk Customers** — a ranked action list where each row carries a *per-customer* SHAP explanation (▲ = pushes risk up) and a concrete next-best-offer from the recommender, with an opt-in AI Agent Layer section for plain-English risk explanations and drafted retention messages (see [AI Agent Layer](#ai-agent-layer)).
- **Customers** — a searchable customer list and per-customer detail page, a capability the platform's original Streamlit dashboard never had.
- **Segments** — churn rate broken out across demographic/account segments.
- **Model Performance** — the model as it actually is: AUC, recall/precision at the chosen threshold, and top feature importances.
- **Pipeline Status** — the orchestration layer exposed to the same audience: batches ingested, batches remaining until the next conditional retrain, model versions trained to date, and an AI-generated plain-English summary of the most recent real retrain.
- **AI Assistant** — a chat interface over the RAG-backed AI Decision Assistant (see [AI Agent Layer](#ai-agent-layer)).

The platform originally shipped this surface as a Streamlit dashboard; it was retired in favor of this Next.js app (see git history for the Streamlit-era screenshots and verification narrative this README used to include).

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

The natural choice for "big data" processing is often Apache Spark. This project deliberately uses **DuckDB** instead — an in-process analytical database that handles millions of rows efficiently with a fraction of Spark's memory footprint, and has a native dbt adapter. The development machine has **8GB RAM total, and frequently under 1GB free** at any given moment (confirmed during this build - see Limitations). Running Spark alongside Postgres, Airflow, and everything else simultaneously would not be reliable on hardware like this. DuckDB delivers most of the analytical performance benefit of a distributed engine without the operational overhead, for workloads — like this one — that don't genuinely require a multi-node cluster. In practice, processing a ~77K-row simulated batch (1/13th of the dataset) through DuckDB takes well under a second.

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
              FastAPI (src/model/api.py)
   POST /predict-churn  POST /recommend  GET /explain-churn/{id}
   GET /overview/*  GET /at-risk  GET /customers  GET /pipeline-status
     (loads latest artifact of each at startup, no live DB dependency
      at request time for predictions; dashboard-facing endpoints read
      customer_360 directly via src/model/dashboard_queries.py)
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

## Repository Structure

```
.
├── dags/
│   └── churn_pipeline.py          # Airflow DAG: ingest -> dbt run/test -> drift check -> conditional retrain
├── src/
│   ├── warehouse.py               # server-side-cursor streaming reads (see Deviations)
│   ├── ingest/
│   │   ├── split_batches.py       # one-time: splits the Kaggle CSV into 13 batch files
│   │   └── batch_loader.py        # picks up next batch, DuckDB clean/validate/engineer, loads to Postgres
│   ├── model/
│   │   ├── train_churn.py         # trains, calibrates, and evaluates the churn classifier, saves versioned artifact
│   │   ├── train_recommender.py   # trains the content-based recommender, saves versioned artifact
│   │   ├── evaluate_recommender.py # leave-one-out offline evaluation vs. popularity/random baselines
│   │   ├── explain_churn.py       # shared per-customer SHAP logic (used by both /explain-churn and /at-risk)
│   │   ├── threshold_analysis.py  # expected-value threshold selection (cost/benefit model)
│   │   ├── dashboard_queries.py   # warehouse/filesystem reads behind api.py's dashboard-facing endpoints
│   │   └── api.py                 # FastAPI app: predictions, dashboard-facing REST endpoints, AI assistant
│   ├── monitoring/
│   │   └── drift.py               # PSI drift detection between batches (see Drift monitoring)
│   └── agents/                    # AI Agent Layer - presentation only, see AI Agent Layer
│       ├── groq_client.py         # shared Groq wrapper: retry/backoff, model tiers, token logging
│       ├── explanation_agent.py   # SHAP values -> plain-English churn explanation
│       ├── outreach_agent.py      # explanation + recommendation -> drafted retention message
│       ├── retrain_summary_agent.py # old vs. new metrics + drift -> plain-English retrain summary
│       └── cache.py               # Postgres-backed cache, keyed by customer + agent + model version
├── frontend/                       # Next.js frontend - the platform's UI, talks to api.py's REST endpoints
│   ├── app/                        # App Router pages: overview, at-risk, customers, segments, model-performance, pipeline-status, assistant
│   ├── components/                 # KpiCard, ChurnBarChart, ShapFactorList, etc.
│   └── lib/                        # api-client.ts (typed fetch per endpoint), types.ts
├── dbt/
│   ├── models/
│   │   ├── staging/                # 5 models: demographics, account, services, usage, churn
│   │   ├── intermediate/           # 3 models: service summary, usage summary, value segment
│   │   └── marts/customer_360.sql  # the unified customer view
│   ├── macros/generate_schema_name.sql
│   ├── dbt_project.yml
│   └── profiles.yml
├── db/
│   └── schema.sql                  # raw_customers, customers_cleaned, ingestion_log, feature_drift, llm_explanations, retrain_summaries DDL
├── data/
│   └── raw/                        # batch_001.csv ... batch_013.csv (gitignored, regenerate via split_batches.py)
├── notebooks/
│   ├── model_dev_offline.py                    # offline model-comparison experiment
│   ├── churn_hyperparameter_sweep.py           # CV hyperparameter search (see Verified state)
│   ├── eda_and_statistical_analysis.ipynb      # significance tests, survival analysis, clustering, SHAP
│   └── threshold_and_business_value.ipynb      # expected-value threshold + calibration analysis
├── tests/
│   ├── fixtures/sample_batch.csv
│   ├── test_ingest.py              # 7 tests: DuckDB cleaning/validation/feature logic
│   ├── test_model.py               # 13 tests: expected-value math, threshold selection, recommender invariants
│   ├── test_drift.py               # 18 tests: PSI correctness, incl. shifts the detector must catch
│   ├── test_evaluate_recommender.py # 20 tests: ranking metrics, leave-one-out splitting, significance tests
│   ├── test_dashboard_queries.py   # pins the CalibratedClassifierCV/.named_steps regression (see Deviations)
│   └── test_agents.py              # 28 tests: AI Agent Layer - prompt grounding, retry/backoff, fallback (Groq mocked)
├── docs/
│   └── images/                     # historical screenshots from the retired Streamlit dashboard
├── report/
│   ├── findings.md                 # business-facing write-up (synthetic-data caveat up front)
│   └── retrain_summaries/          # AI-generated, one .md per real retrain (written by summarize_retrain)
├── docker/
│   ├── Dockerfile.airflow
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   ├── Dockerfile.mlflow
│   ├── Caddyfile                   # self-hosted TLS (see Production hardening)
│   ├── requirements-airflow.txt
│   └── init-multi-db.sh
├── scripts/
│   └── migrate_to_neon.sh          # one-time: copies customer_360 + operational tables to a hosted Postgres
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt                # pipeline + serving + CI
├── requirements-notebooks.txt      # analysis-only extras (see Statistical & analytical depth)
├── pytest.ini
├── .env.example
├── .gitignore
└── LICENSE
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

37 dbt tests (`not_null`, `unique`, `accepted_values`) run across all three layers — all passing as of the last verified run (see "Verified state" below).

## Statistical & analytical depth

Beyond the production pipeline, `notebooks/eda_and_statistical_analysis.ipynb` is a fully executed (not just written) analysis notebook that goes past descriptive segment charts into actual statistical methodology:

- **Significance testing**: chi-square tests (with Cramer's V effect size) for every categorical feature against churn, and Mann-Whitney U tests (with rank-biserial effect size) for every numeric feature. Finding: `contract` and `tenure_bucket` are genuinely associated with churn; `gender`, `education`, `marital_status`, and `payment_method` are **not** statistically significant at all — no demographic "churn persona" is supported by this data.
- **Statistical vs. practical significance**: at 300K+ rows, several features reach p < 0.05 with a practically negligible effect size (e.g. `monthlycharges`, hazard ratio ≈ 0.998 per dollar) — called out explicitly rather than reported as a bare "significant!" p-value.
- **Correlation & multicollinearity**: a correlation heatmap plus Variance Inflation Factors across key numeric features.
- **Survival analysis**: Kaplan-Meier curves (overall and by contract type) and a Cox Proportional Hazards model — modeling *time to churn* directly, which the binary classifier discards. `is_month_to_month` carries a hazard ratio of ~2.77 (holding other factors constant); concordance index 0.617. Kaplan-Meier: 94.7% of customers still active at 12 months tenure, 90.2% at 24, 82.1% at 48.
- **Unsupervised customer segmentation**: K-means clustering on profile features (age, income, tenure, charges, satisfaction, usage, service count), with an elbow/silhouette analysis to choose k and a PCA projection to visualize it. Reported honestly: silhouette scores are modest (~0.14–0.16), meaning the natural cluster structure is soft, not sharply separated — stated plainly rather than overclaimed.
- **SHAP explainability**: TreeExplainer on the production LightGBM model, both as a global summary plot and individual waterfall plots for specific high-risk and low-risk customers. This same SHAP logic is also used live in the **frontend's At-Risk Customers view** (`compute_shap_details` in `src/model/explain_churn.py`) — replacing an earlier global-feature-importance heuristic with real per-customer explanations (bounded to the displayed rows, not the full 1M-row table, for memory reasons).

### Re-running the analysis

The notebooks need libraries the pipeline itself doesn't (statsmodels, lifelines, xgboost, imbalanced-learn, matplotlib, seaborn). Those live in a **separate** requirements file, deliberately: `requirements.txt` is installed into the API image and every CI run, neither of which import any of them.

```bash
pip install -r requirements.txt -r requirements-notebooks.txt
jupyter lab notebooks/
```

Versions there are pinned to the ones the committed outputs were produced with, so a re-run reproduces the numbers in `report/findings.md` rather than approximately reproducing them.

See the notebook itself for full output, and `report/findings.md` for the business-facing summary of these findings.

## Drift monitoring

Every batch that arrives is scored against a fixed baseline batch using the **Population Stability Index**, the standard drift metric in credit-risk and churn modeling:

```
PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

PSI < 0.10          stable        - no action
0.10 <= PSI < 0.25  moderate      - investigate
PSI >= 0.25         significant   - retrain
```

19 features are monitored (14 numeric, 5 categorical). Results are written to `public.feature_drift` by the DAG's `detect_feature_drift` task and surfaced on the frontend's Pipeline Status view.

This turns retraining into a two-trigger decision rather than a fixed cadence:

| Trigger | Condition | Rationale |
| --- | --- | --- |
| Cadence | every 3rd batch | the model never silently goes stale |
| Evidence | any feature PSI ≥ 0.25 | the incoming distribution no longer matches the training distribution |

Three implementation choices worth calling out, because they are where naive PSI implementations go wrong:

- **Bin edges come from the reference distribution only**, never recomputed per batch. Re-binning on the new data makes every batch look identical to itself and hides precisely the shift being measured.
- **NULLs are their own bucket**, not dropped. A feature whose missing rate jumps from 3% to 40% has drifted in the way that matters most operationally; dropping NULLs scores that as perfectly stable.
- **The outer bin edges are open** (`-inf`, `+inf`), so values beyond the reference range — the most obvious kind of drift — land in the end bins instead of being silently discarded as out-of-range.

**The honest result on this dataset: no drift, as expected.** All 13 batches are sequential slices of one pre-shuffled source file, so the maximum PSI observed across every feature and every batch pair is **0.0006** — three orders of magnitude below the "investigate" threshold. That is the correct answer for this data, not a broken detector. Evidence that the detector does fire lives in `tests/test_drift.py` (18 tests), which injects real shifts and asserts they are caught: mean shifts, variance shifts with an unchanged mean, missingness jumps, unseen categorical levels, and out-of-range values.

Run it manually against any ingested batch:

```bash
python -m src.monitoring.drift batch_013.csv
```

## Probability calibration

`class_weight="balanced"` is what makes the churn model's *ranking* work under a ~10%-imbalanced target, but it does so by inflating minority-class scores, so the raw output is not a probability at all — mean predicted score was **0.41** against an actual churn rate of **0.10**, a 4.1x overstatement, and the model's own Brier score (0.196) was *worse* than simply always predicting the base rate (0.090).

`train_churn.py` now fixes this with a three-way split (train / calibration / test) and `CalibratedClassifierCV` (sigmoid / Platt scaling) fit on the held-out calibration split — never the training or test data, so nothing leaks into the reported metrics:

| | Brier score | Mean predicted | Actual rate |
| --- | --- | --- | --- |
| Uncalibrated | 0.196 (worse than the 0.090 baseline) | 0.408 | 0.100 |
| **Calibrated (sigmoid)** | **0.087** (beats the baseline) | **0.100** | 0.100 |

Sigmoid rather than isotonic: both reached the same Brier score in testing, but sigmoid is a strictly monotonic two-parameter fit, so it leaves ROC AUC exactly unchanged (0.6693 either way) and is far less prone to overfitting a ~15K-row calibration split than isotonic's free-form step function.

**Calibration changes what the numbers mean, not what the model does.** Because sigmoid scaling is monotonic, ranking and the operating point are unaffected — precision (0.160) and recall (0.600) at the chosen threshold are the same before and after. What changes is that the decision threshold now sits at **0.105** instead of **0.451**, because it is now a real probability near the ~10% base rate rather than an arbitrary score compressed toward 0.5. One consequence worth knowing before you go looking for it: **a naive 0.5 threshold on the calibrated model now flags zero customers** — not a bug, but exactly the failure mode calibration exists to expose. The full before/after, including the expected-value threshold analysis recomputed on the calibrated scores, is in `report/findings.md` Section 6.

This also broke something non-obvious enough to be worth flagging on its own: `CalibratedClassifierCV` has no `.named_steps`, which the dashboard's SHAP explanations and feature-importance fallback both depend on. See [Deviations](#deviations-from-the-original-spec-and-why) for how that was caught and fixed.

## Recommender evaluation

Every other model in this project is measured against a baseline on a held-out split; until now the recommender wasn't — its metadata recorded only structural facts (profile count, neighbor count). `src/model/evaluate_recommender.py` closes that gap with a leave-one-out protocol: hide one service a customer genuinely owns, present them to the model as if they didn't own it, and check where the hidden service lands in the ranking of everything else they don't own. Three rankers are scored on identical splits, over customers held **entirely outside** the k-NN reference set (scoring the model on profiles it was fitted on would measure memorization, not generalization):

| Ranker | Hit@1 | Hit@2 | Hit@3 | MRR |
| --- | --- | --- | --- | --- |
| **k-NN (production)** | **0.519** | 0.737 | 0.878 | **0.703** |
| Popularity baseline | 0.483 | 0.732 | 0.891 | 0.686 |
| Random | 0.245 | 0.478 | 0.681 | 0.498 |

**The honest result: k-NN beats popularity, but by a small margin, and popularity is a genuinely hard baseline to clear here** — 8 services with skewed adoption rates (internet 84.8% down to device protection 29.6%) leave little room for a naive ranker to be bad. A table alone can't say whether +2.5% MRR is signal or noise, so the module runs two significance tests rather than reporting the gap and moving on: a paired bootstrap (2,000 resamples, customer-level) puts the MRR gap at **+0.0170, 95% CI [+0.0105, +0.0237]**, and an exact McNemar test on Hit@1 gives **p = 2.3×10⁻⁹**. Both agree the edge is real, not sampling noise — but it is a small edge, reported as one, not rounded up to "the model works great" or down to "personalization doesn't matter here."

```bash
python -m src.model.evaluate_recommender
```

Full detail, including why customers with fewer than 2 candidate services are excluded from evaluation (a guaranteed hit that would inflate every ranker equally), is in `report/findings.md` Section 5a.

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

Written to `report/retrain_summaries/retrain_summary_20260910T154013Z.md` and to the `retrain_summaries` Postgres table, and shown on the frontend's Pipeline Status view. The regression itself is expected noise (see [Verified state](#verified-state) for why: both artifacts trained on the same 150K-row sample with the same random seed, so this reflects sampling variance in the calibration/test split, not a real capability drop) - included here specifically *because* it's the honest case, not the flattering "essentially unchanged" one from an earlier direct test.

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

## How to run

### 1. Prerequisites

- Docker Desktop
- A `kaggle.json` API token at `~/.kaggle/kaggle.json` (get one from kaggle.com → Account → Create New API Token) — only needed once, to download the dataset
- Copy `.env.example` to `.env` and fill in real values (a Postgres password and Airflow admin password at minimum)
- Optional: a free `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys) for the [AI Agent Layer](#ai-agent-layer) — everything else works without it

### 2. One-time data setup

```bash
kaggle datasets download -d isandeep06/customer-churn-prediction-dataset-1m -p data/raw_download --unzip
python src/ingest/split_batches.py   # writes data/raw/batch_001.csv ... batch_013.csv
```

### 3. Start the warehouse + orchestration

```bash
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler
```

Airflow UI: http://localhost:8081 (login from `.env`'s `AIRFLOW_ADMIN_USER`/`AIRFLOW_ADMIN_PASSWORD`; 8081, not the default 8080 — see Deviations). Trigger the `churn_pipeline` DAG manually — each run ingests one more batch (simulating one weekly arrival), then runs dbt, then conditionally retrains.

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

`api` and `frontend` are behind Docker Compose's `serving` profile, so a plain `docker compose up` (warehouse + orchestration only) doesn't also pay for containers you may not be actively using — deliberate, given the RAM constraint. The Next.js app under [`frontend/`](frontend/) is the platform's UI — talking to `api`'s dashboard-facing REST endpoints (`/overview/*`, `/at-risk`, `/customers`, `/pipeline-status`, `/model-history`) instead of reading Postgres directly.

To run the frontend outside Docker for local development (hot reload):

```bash
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

### 6. Public demo deployment (optional)

Both the API and the frontend can run reachable from anywhere — [Neon](https://neon.tech) (free Postgres) plus [Render](https://render.com) (free Docker-based web services) — without cloning the repo or running Docker locally. Deliberately scoped to just these two services — Airflow stays local-only (it's an admin/orchestration tool with a login, not something a portfolio visitor needs to see live), and the warehouse the API reads is a **read-mostly copy**, not the live pipeline.

1. **Create a free [Neon](https://neon.tech) project.** Copy its connection details (host, database, user, password — Neon shows these as one connection string; split it into the pieces below).
2. **Migrate the data** the API actually reads (`marts.customer_360` plus the small operational tables — deliberately *not* the raw/staging pipeline tables, which the serving layer never queries and would roughly double the transfer):
   ```bash
   docker compose up -d postgres   # local warehouse must be running
   ./scripts/migrate_to_neon.sh "postgresql://user:pass@host/dbname?sslmode=require"
   ```
   Run for real against a live Neon database, not just a local stand-in: all 1,000,000 `customer_360` rows, 228 drift rows, 13 ingestion-log rows round-tripped correctly (213MB transferred), re-verified with direct queries against Neon itself afterward.
3. **Push this repo to GitHub** (already done — [github.com/LyHenglong/Customer-Intelligence-Platform](https://github.com/LyHenglong/Customer-Intelligence-Platform)).
4. **Create a [Render](https://render.com) account**, then **New → Blueprint**, and point it at this GitHub repo. [`render.yaml`](render.yaml) is a ready-to-use [Render Blueprint](https://render.com/docs/blueprint-spec) that Render reads from the repo root automatically, creating both `telecom-churn-api` and `telecom-churn-frontend` as standalone web services from `docker/Dockerfile.api`/`docker/Dockerfile.frontend`. Render was picked over alternatives (Fly.io, Railway, Vercel for the frontend) mainly to keep one platform for both services, with a free tier that needs no credit card and supports Dockerfile-based web services directly.
5. **Fill in `telecom-churn-api`'s env vars** Render prompts for (`POSTGRES_HOST/DB/USER/PASSWORD`) with the same Neon project from step 1. `GROQ_API_KEY` is optional — leave it out unless you want a public visitor able to trigger real (rate-limited, cost-bearing) calls against your own Groq quota; see the [AI Agent Layer](#ai-agent-layer) guardrails section for what stays intact either way (the platform runs completely normally without it, just without the AI prose). Leave `CORS_ALLOWED_ORIGINS` and `telecom-churn-frontend`'s `NEXT_PUBLIC_API_URL` on their placeholder for now — neither service's real URL exists yet.
6. **Copy each deployed service's URL** (Render shows these after the first successful deploy — something like `https://telecom-churn-api.onrender.com` and `https://telecom-churn-frontend.onrender.com`).
7. **Set `CORS_ALLOWED_ORIGINS`** on `telecom-churn-api` to the frontend's URL, and **`NEXT_PUBLIC_API_URL`** on `telecom-churn-frontend` to the API's URL, then **manually redeploy `telecom-churn-frontend` specifically** — `NEXT_PUBLIC_API_URL` is inlined into the browser bundle at build time (see `docker/Dockerfile.frontend`'s comment), so an env var update alone doesn't take effect until the next build.

`src/warehouse.py`'s `get_pg_conn()` connects with `sslmode="prefer"` (configurable via `POSTGRES_SSLMODE`), so the identical code reaches both the local Docker Postgres (no SSL) and Neon (SSL required) without an environment-specific branch — see Deviations for the real bugs this deployment path surfaced along the way.

Two things worth knowing before relying on this:
- **Free-tier cold starts**: Render's free web services spin down after ~15 minutes idle; the first request after that pays a 10-50s cold-start delay to spin back up. Expected behavior on the free plan, not a bug — a visitor's first request after a quiet period will just be slow, not broken.
- **No MLflow alongside the API**: this deployment has no `mlflow` service reachable from it, so `src/model/registry.py` falls back to the glob-latest-by-timestamp artifact already baked into the Docker image at build time — correct, and by design (see that module's own docstring), but it does mean this deployment always serves whatever was newest in `models_store/` at the image's last build, not a live-promoted model.

### dbt directly

```bash
cd dbt
export DBT_PROFILES_DIR=$(pwd)   # profiles.yml lives in the project, not ~/.dbt/
dbt run
dbt test
```

### Tests

```bash
pytest tests/ -v
```

95 tests in this table's own subset, all operating on synthetic in-memory fixtures — no database, no Docker, no trained model artifact, and (for the AI Agent Layer) no real Groq API key required, which is what lets the GitHub Actions workflow run them on a clean checkout. The full suite (`pytest tests/`) is larger still - it also covers the AI Agent Layer's RAG/tool-calling internals and the FastAPI dashboard-facing endpoints, not tabulated individually here.

| File | Tests | Covers |
| --- | --- | --- |
| `test_ingest.py` | 7 | DuckDB type coercion, quality-gate filtering, engineered-feature logic |
| `test_model.py` | 13 | expected-value math, threshold selection, recommender invariants |
| `test_drift.py` | 18 | PSI correctness, and the shifts the detector is required to catch |
| `test_evaluate_recommender.py` | 20 | ranking metrics, leave-one-out splitting, bootstrap/McNemar significance tests |
| `test_dashboard_queries.py` | 9 | `column_importances`/SHAP behave correctly on both a raw pipeline and a `CalibratedClassifierCV` wrapper, plus the rest of the dashboard-facing warehouse reads |
| `test_agents.py` | 28 | AI Agent Layer: prompt grounding in real SHAP/metrics data, retry/backoff on rate limits, graceful fallback on failure (Groq fully mocked) |

## Production hardening

Everything below is real, working configuration and code - not aspirational. None of it requires provisioning a new paid service to be useful; each piece degrades gracefully to "off" if its optional env var is unset, same convention as the AI Agent Layer and the model registry above.

**TLS.** Automatic and free wherever this project actually deploys: Render (the API and the Next.js frontend, see step 6 above) terminates TLS at its own edge - no certificate config needed. [`docker/Caddyfile`](docker/Caddyfile) + the `caddy` service (`docker compose --profile production up -d caddy`) covers the one case that doesn't: self-hosting this stack directly on your own server with a real domain. Needs `API_DOMAIN`/`FRONTEND_DOMAIN` set to real DNS A records pointing at the host - Caddy handles Let's Encrypt issuance and renewal automatically from there.

**Secrets.** No dedicated secrets-manager tool (Vault, Doppler, etc.) - deliberately, matching this project's existing "avoid unnecessary infrastructure" pattern (followed elsewhere for the same reason - e.g. no Redis, no separate queue). Every real deployment surface already has its own encrypted secret store: Render's env var UI (`sync: false` entries in `render.yaml` prompt for these rather than committing them), and locally, `.env` (gitignored, never committed - verified: `git log --all -- .env` shows nothing). Nothing in this repo hardcodes a credential; every `POSTGRES_PASSWORD`/`GROQ_API_KEY`/etc. is read from the environment (`os.environ.get(...)` throughout, e.g. `src/warehouse.py`).

**CI/CD.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (pre-existing) runs the test suite on every push/PR. [`.github/workflows/cd.yml`](.github/workflows/cd.yml) builds and pushes the `api`/`frontend` Docker images to GitHub Container Registry after CI passes on `main` - tagged both `latest` and the commit SHA, so a specific deploy is always traceable/rollback-able. It deliberately triggers off CI's own success (`workflow_run`) rather than duplicating the test job, so there's one source of truth for "does this pass."

**Horizontal scaling.** Verified `src/model/api.py` is safe to run as multiple replicas before writing any scaling config: it never writes to local disk at request time (grepped for it - nothing outside the read-only `models_store/` mount), and everything it does persist (AI traces, the LLM explanation cache, drift results) already goes to the shared Postgres database, not process-local state. `render.yaml` documents the one-line change (`numInstances`) to turn this on - commented out by default since it needs a paid Render plan, not the free tier this repo otherwise assumes.

**Backups.** [`dags/backup_warehouse.py`](dags/backup_warehouse.py) - a real, `@daily`-scheduled DAG that `pg_dump`s the warehouse, gzips it into `./backups/`, and prunes anything older than `BACKUP_RETENTION_DAYS` (default 14). It skips itself automatically (checks `POSTGRES_HOST`) against anything that isn't the local docker-compose `postgres` service, because a hosted provider like **Neon already provides automated backups/point-in-time recovery natively** - running a second, unwatched local backup mechanism against a Neon-backed deployment would be redundant, not extra safety.

**Alerting.** [`src/monitoring/alerting.py`](src/monitoring/alerting.py) posts to a Slack incoming webhook (`SLACK_WEBHOOK_URL`) - wired into two real triggers: `detect_feature_drift` (any batch with a "significant"-severity feature, see [Drift monitoring](#drift-monitoring)) and every Airflow task's `on_failure_callback` (fires once retries are exhausted, both `churn_pipeline` and `backup_warehouse`). Unset means every call is a silent no-op - verified with `tests/test_monitoring_alerting.py`, which also confirms a Slack/network failure is caught and logged rather than propagated (an alerting problem must never take down the pipeline step it's reporting on). Create a webhook at [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks) to enable it.

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
| AI Agent Layer | Groq (`openai/gpt-oss-120b` / `-20b`), presentation only - see [AI Agent Layer](#ai-agent-layer) |
| Model serving | FastAPI |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, Recharts |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest, dbt tests |

## Verified state

Everything below was actually run and checked during the build, not just written and assumed to work:

- **Postgres**: up, reachable, both databases (`warehouse`, `airflow_meta`) and all three dbt schemas created.
- **Ingestion**: ran on 4 batches total (`batch_001`–`batch_004`, 2 manually + 2 via the Airflow DAG). Verified row counts, type coercion, boolean conversion, NULL preservation (null rates in `customers_cleaned` matched the source file's null rates within rounding), and feature engineering values by hand.
- **dbt**: `dbt run` builds all 9 models; `dbt test` passes all 37 tests, re-verified after every batch. `customer_360` grew in exact lockstep with batches ingested (76,924 x n), confirming append-not-replace behavior, through all 13 batches to the full **1,000,000 rows**.
- **Churn model**: originally RandomForest, trained twice via the DAG (76,924-row baseline and a 230,772-row auto-retrain), then replaced after a dedicated model-improvement investigation (`notebooks/model_dev_offline.py`) compared RandomForest/XGBoost/LightGBM/Logistic Regression, SMOTE vs. `class_weight`, and ran 5-fold cross-validation on the full 1,000,000-row dataset. **LightGBM won by a small, cross-validation-confirmed margin** (AUC 0.683 ± 0.002 vs RandomForest's 0.677 over the full 1,000,000 rows) and is now the production model, with a business-chosen decision threshold instead of the default 0.5.

  Different numbers appear in this project and they are not interchangeable — the deployed model's own test-set score is the one quoted in the UI:

  | | Rows | ROC AUC | Purpose |
  | --- | --- | --- | --- |
  | Model-selection experiment | 1,000,000 (5-fold CV) | 0.683 ± 0.002 | choosing between model families |
  | Hyperparameter sweep, winning config | 200,000 (4-fold CV) | 0.678 ± 0.003 | choosing hyperparameters |
  | **Deployed artifact** (`20260923T040125Z`) | 150,000 | **0.666** | what actually serves predictions |

  The deployed model trains on a 150,000-row sample because the retrain task runs in-process under Airflow's LocalExecutor on a 3.8GB Docker VM and was SIGKILLed at full scale (see Deviations); AUC was measured to be flat at 0.65–0.68 across sample sizes, so the cap costs little. Its full test-set metrics: precision 0.157, recall 0.600, F1 0.249, threshold 0.104. See "Limitations" and `report/findings.md` — this is still a modest model, and that's discussed honestly, not hidden.

- **Hyperparameter tuning**: the model-family audit above never varied LightGBM's own hyperparameters — it ran throughout on the defaults chosen when the family was picked. [`notebooks/churn_hyperparameter_sweep.py`](notebooks/churn_hyperparameter_sweep.py) scored 9 configurations by 4-fold cross-validation on identical folds, with the adoption rule fixed **before** any numbers were seen: beat the baseline by more than one baseline fold-to-fold standard deviation, or change nothing. `max_depth=4, n_estimators=600, learning_rate=0.05` won at 0.6778 ± 0.0026 against the baseline's 0.6651 ± 0.0014 — roughly 9x the baseline std, and monotone rather than a lucky fold: depth 8 barely moved (+0.0034) and removing the depth limit entirely *hurt* (−0.0118), so the previous configuration was spending capacity memorising noise in a weak-signal target. Retrained into production it transferred almost exactly as predicted (+0.0131 observed vs +0.0127 predicted): AUC 0.6529 → **0.6660**, precision 0.1514 → **0.1573** at unchanged recall (0.600, fixed by the threshold rule), F1 0.2418 → **0.2493**. This narrows the gap to the 0.65–0.68 band's ceiling rather than escaping it — see `report/findings.md` §2a.
- **Real-data signal benchmark**: the production model's ~0.67 AUC is a property of the synthetic dataset, not the modelling code, and that is now measured rather than asserted. Scoring each feature alone shows **27 of 38 are individually indistinguishable from noise** (AUC < 0.52), with the best single feature — `contract` — carrying only 0.614. [`src/model/train_churn_telco.py`](src/model/train_churn_telco.py) runs the *identical* method (same three-way split, calibration, threshold rule and tuned hyperparameters) against the real IBM Telco dataset and reaches **0.8347 ± 0.0115** 5-fold CV / **0.8269** held-out, with precision at recall ≈ 0.60 going from 0.157 to **0.598**. Same code, +0.16 AUC — the data was the limitation. It is a benchmark and is deliberately **not served**: it is 7,043 rows on a different schema, named `telco_churn_*` precisely so the production `churn_model_*` glob cannot pick it up (pinned by a regression test). See `report/findings.md` §2b. Worth knowing: ~0.83 is near the practical ceiling for churn even on good real data; an AUC above 0.90 on churn usually means target leakage rather than a better model.
- **Recommender**: only ever recommends services the customer doesn't already have, ranked by neighbour popularity, and this is now backed by a real offline evaluation rather than a spot-checked example — see [Recommender evaluation](#recommender-evaluation). It is retrained by the DAG on the same trigger as the churn model — previously it wasn't retrained at all, so its artifact had aged 9 hours behind the churn model and was still built from 2 batches' worth of customers rather than 13. Retraining at the full 1,000,000-row scale was verified inside the `airflow-scheduler` container: it samples a bounded 150,000-profile reference set, produces a **9.9MB** artifact (down from 17MB), and the API loads it at **206MB of its 400MB limit** with `/recommend` coverage still 40/40 on random real customers.
- **FastAPI**: both endpoints tested with a real customer's data pulled from the warehouse. `/recommend` coverage was re-verified after the warehouse-fallback fix on 40 randomly sampled real customer IDs — 40/40 returned recommendations, against ~15% before the fix — while an ID that genuinely doesn't exist still returns 404 and the container stayed at 258MB of its 400MB limit. `/predict-churn` returned a probability consistent with that customer's actual (held-out) label; `/recommend` returned 3 ranked un-subscribed services; a 404 for an unknown `customer_id` was also verified. Re-tested after the DAG's auto-retrain to confirm the API correctly picks up and serves the newest versioned artifact — this is also what surfaced the `dill` cross-environment issue above.
- **Streamlit dashboard** *(retired - this platform's original UI, since replaced by the Next.js frontend under [`frontend/`](frontend/))*: verified headlessly via Streamlit's `AppTest` runner *and* by driving the running container with headless Chromium, capturing all five views (the historical screenshots this produced were removed from the repo along with the retired app - see git history for them). The browser pass found two layout defects `AppTest` structurally could not — a KPI truncated to `$26,051...` and a clipped SHAP column — both since fixed. Peak container memory during a full five-view capture: 652MB against its 2GB limit.
- **pytest**: 63/63 pass locally, and the same `pytest tests/` command is green on GitHub Actions — most recently on commit `7508e15`, the commit that added the recommender-evaluation and calibration-regression tests, with the dependency-install step (lightgbm, shap, dill) succeeding on a clean Ubuntu runner. Covers ingestion logic, expected-value/threshold math, recommender invariants, PSI drift, recommender evaluation metrics, and the calibration/SHAP artifact-shape regression.
- **Probability calibration**: verified with a genuine train/calibration/test three-way split (never reusing test data for calibration). Brier score improved from 0.196 (worse than the 0.090 always-base-rate baseline) to 0.087 (better than it); mean predicted probability moved from 0.408 to 0.100 against an actual rate of 0.100. AUC confirmed unchanged (0.6693 before and after, since sigmoid scaling is monotonic). The expected-value threshold analysis was fully recomputed on the calibrated model at the same 200,000-row test-set scale as the original analysis (not spot-checked) — see `report/findings.md` Section 6.
- **Recommender evaluation**: 4,918 evaluation customers, drawn entirely outside the k-NN reference set, scored with leave-one-out hit-rate/MRR against popularity and random baselines. k-NN's edge over popularity (+2.5% MRR) confirmed significant by both a paired bootstrap (95% CI entirely above zero) and an exact McNemar test (p = 2.3×10⁻⁹) — not just eyeballed off a metrics table.
- **The SHAP-column regression from adding calibration was caught before this was written up, not after.** Re-reading the At-Risk Customers screenshot post-retrain showed an empty "Risk factors (SHAP)" column with no exception anywhere in the logs — `column_importances` was silently swallowing a new `AttributeError` inside a `try/except` written for a different, legitimate reason. Fixed by saving the pre-calibration pipeline as `base_pipeline` in the artifact; re-verified by re-capturing the same screenshot and confirming the column populated again, and pinned with 5 new tests that would fail if this regressed.
- **Drift detection**: computed for all 12 non-baseline batches against the `batch_001` baseline, both from the host and from inside the `airflow-scheduler` container (proving the DAG runtime can actually execute it). Maximum PSI observed across every feature and every batch pair: **0.0006** — a correct negative result on a pre-shuffled source file, not a silent failure. The 18 tests in `tests/test_drift.py` inject genuine shifts and confirm the detector fires on them.
- **The drift task was verified in a real DAG run**, not just by parsing. `batch_013` was removed from `ingestion_log`/`customers_cleaned`/`raw_customers` and re-ingested through a live triggered run: all 8 tasks succeeded in 97 seconds, and the branch logged its decision from both inputs — `batches: 13 (every 3 -> due=False); drift: max PSI 0.0003 -> detected=False` → `Branch into skip_retrain`, correctly skipping the retrain. Warehouse verified back at 1,000,000 rows afterwards.
- **Airflow — fully verified end-to-end**, not just built. Two full DAG runs against the live webserver + scheduler (not `airflow tasks test` shortcuts): run 1 ingested `batch_003` and correctly triggered `retrain_churn_model` (3rd batch landed → branch fired, produced a new versioned artifact alongside the original, neither overwritten); run 2 ingested `batch_004` and correctly took the `skip_retrain` branch instead (4th batch, not a multiple of 3). All 7 tasks in both runs ended in `success` (or the intentionally-`skipped` branch). `customer_360` grew exactly as expected across both runs, confirmed by direct row-count query, not just DAG "green" status.
- **AI Agent Layer — all three agents run against real data from this running platform, not just their 28 mocked tests.** `explain_churn()` verified via a live `curl` to `/explain-churn/CUST0000269643` (`source: "llm"`, a real 32.6% probability, real SHAP factors) and via the dashboard's rebuilt container. `draft_outreach()` verified the same way, and `mentions_service()` confirmed the real draft actually named the real recommended service ("Internet Service"), not a hallucinated one. `summarize_retrain()` was verified as an actual Airflow task in a live DAG run - two batches (`batch_012`, `batch_013`) were reset and re-ingested to force a real retrain, `summarize_retrain` ran and correctly wrote both `report/retrain_summaries/retrain_summary_20260910T154013Z.md` and a Postgres row, and the warehouse was restored to its full 13-batch state afterward. The caching layer was verified to actually skip redundant calls (~4.5s on a miss, ~0.02s on a hit), and the retry/backoff guardrail was exercised for real, not just mocked - generating AI content for 15 real customers in one session genuinely hit Groq's free-tier rate limit mid-run and recovered cleanly (see Deviations).
- **The Streamlit Cloud + Neon deployment path is verified against a real, live Neon database, not just a local stand-in.** Once the user created the free Neon project and shared its connection string, `scripts/migrate_to_neon.sh` ran for real: `CREATE SCHEMA`, `COPY 1000000` (`customer_360`), `COPY 228` (`feature_drift`), `COPY 13` (`ingestion_log`) all succeeded against the live endpoint, then re-verified with direct row-count and spot-check queries run against Neon itself (the exact customer used in the AI Agent Layer's examples round-tripped correctly, and the churn rate matched exactly: 9.92%). The dashboard was then run for real with its Postgres env vars pointed entirely at Neon (`POSTGRES_SSLMODE=require`, no local Postgres involved at all) and rendered all 1,000,000 real customers correctly end to end.
- **That same test surfaced a real, honest limitation: first-load latency against Neon's free tier depends heavily on network distance.** Measured directly from this development machine: the dashboard's full streaming query (1,000,000 rows × 39 columns) took roughly **8 minutes** against Neon's `us-east-2` region, versus low single-digit seconds against the local Docker Postgres it normally runs on — isolated and confirmed with a standalone timed query (100,000 rows in 46.3s, linearly projecting to ~463s at 1M), not just an impression from watching a slow page load. This machine is far from `us-east-2`; **Streamlit Community Cloud itself runs on US infrastructure, meaningfully closer to Neon's US regions**, so the real first-load time once actually deployed there is very likely much lower than this worst-case measurement - but that specific number remains genuinely unverified, since it requires the live Streamlit Cloud deployment to exist. Streamlit's own `@st.cache_data(ttl=600)` means this cost is paid once per 10-minute window, not on every click, which matters for a demo session but not for the very first visitor after a cold start.

## Deviations from the original spec, and why

- **Postgres host port changed to 5433.** This machine already runs a native Windows PostgreSQL 17 service on port 5432. Docker's port-forwarding on Windows silently routed `127.0.0.1:5432` connections to that native service instead of the container, causing password-auth failures that looked like a credentials bug but weren't. Remapped the container to `5433:5432` on the host; internal container-to-container traffic (Airflow, dashboard) is unaffected since it uses the Docker network's `postgres:5432`.
- **Postgres `shm_size` raised to 256MB.** The default 64MB Docker shared-memory limit caused `customer_360`'s 7-way join to fail with "could not resize shared memory segment" once the table crossed ~150K rows. This is a known Docker-Postgres gotcha at any nontrivial join size, not specific to this dataset.
- **Added a custom `generate_schema_name` dbt macro.** Without it, dbt's default behavior prefixes custom schemas with the profile's target schema, silently writing to `public_staging`/`public_intermediate`/`public_marts` instead of the `staging`/`intermediate`/`marts` schemas created in `db/schema.sql` — two parallel, disconnected sets of schemas. Fixed by overriding the macro to use the configured schema name exactly.
- **Removed an explicit `sqlalchemy` pin from the Airflow image's requirements.** Pinning `sqlalchemy==2.0.36` force-upgraded the version Airflow 2.10.3 ships with, breaking its ORM (`TaskInstance` mapping errors) and, as a downstream effect, silently dropping the `users` CLI command group.
- **Switched `train_churn.py`/`train_recommender.py`/the dashboard off SQLAlchemy entirely, onto raw `psycopg2` connections.** Even after removing the pin above, Airflow's bundled SQLAlchemy (1.4.54) doesn't pair with pandas 2.2.3's `read_sql` "is this a SQLAlchemy connectable" detection — it silently fell back to a legacy DBAPI path and called `.cursor()` on an `Engine`/`Connection` object that doesn't have one (`AttributeError`). Passing a raw `psycopg2` connection instead sidesteps the detection entirely and works identically in and out of Docker (pandas explicitly supports arbitrary DBAPI2 connections as a fallback path, just with a benign "consider using SQLAlchemy" warning).
- **Added `PYTHONPATH=/opt/airflow` to the Airflow containers.** `docker-compose.yml` bind-mounts `./src` to `/opt/airflow/src`, but Airflow doesn't add its own working directory to `sys.path` by default, so DAG task code's `from src.ingest.batch_loader import ...` raised `ModuleNotFoundError: No module named 'src'` until this was set explicitly.
- **Airflow webserver remapped to host port 8081, not 8080.** An unrelated pre-existing project on this machine ("careerintelligence") already had its own Airflow webserver bound to 8080.
- **Added `dill` as an explicit dependency in `requirements.txt`.** A churn model retrained by the Airflow DAG failed to load in a separately-built environment (`ModuleNotFoundError: No module named 'dill'`) even though nothing in this project imports `dill` directly. Cause: Airflow's own dependency tree imports `dill`, which has the side effect of extending Python's global pickle dispatch table for that process; `joblib.dump()` of a model trained inside an Airflow task inherits that and can end up needing `dill` to unpickle elsewhere. Installing it everywhere an artifact might be *loaded* (not just where it's trained) fixes this regardless of which environment produced a given model version.
- **Ingestion writes two Postgres tables per batch (`raw_customers`, `customers_cleaned`), and dbt's staging layer reads from `customers_cleaned`, not `raw_customers`.** The spec's wording ("staging models: cleaned, typed versions of raw tables") could be read as dbt staging doing the primary cleaning. Instead, DuckDB does mechanical cleaning (type coercion, invalid-row filtering, standardizing booleans) once per batch at ingestion time — cheaper to do close to the data on arrival — and dbt staging's job is reshaping already-clean data into the staging/intermediate/marts layers, plus documentation and tests. This avoids doing the same type-coercion work twice.
- **Airflow's DAG has one `ingest_next_batch` task, not three separate "ingest / DuckDB / load" tasks.** `batch_loader.py` does all three as a single Postgres transaction by design (so a failed/retried batch never leaves the warehouse half-loaded). Splitting that into separate Airflow tasks would mean persisting intermediate DuckDB output between tasks purely to match a step count, adding fragility for no real benefit.
- **13 batches, not 10–15's midpoint.** `1,000,000 / 13 ≈ 76,924` rows/batch; chosen so that `RETRAIN_EVERY_N_BATCHES=3` produces 4 retraining events across a full run (batches 3, 6, 9, 12) — a clean demonstration of the periodic-retrain mechanism.
- **DAG is manually triggered (`schedule=None`), not on a wall-clock schedule.** A DAG run represents one simulated batch "arriving"; a real time-based schedule would misrepresent what's actually a replay of static historical data.
- **Churn model switched from RandomForest to LightGBM, plus a business-chosen decision threshold (0.451 at the time, later 0.105 once probability calibration was added — see below — not 0.5 either way).** The threshold is re-derived at every retrain as the highest cut-off still meeting the recall ≥ 0.60 floor, so it moves with the data rather than being a constant; it is saved inside the artifact and read back by both the API and the dashboard. Mid-build, Docker Desktop's WSL2 backend crashed after the host machine's C: drive filled to 0 bytes free during earlier image builds, corrupting the VM's disk mid-write and requiring a full machine restart to recover. While Docker/Postgres were unavailable, that downtime was used to run a dedicated model-improvement investigation (`notebooks/model_dev_offline.py`) entirely offline — reconstructing `customer_360`-equivalent data straight from the local batch CSVs via `batch_loader.py`'s own DuckDB logic, no database required. The investigation compared RandomForest/XGBoost/LightGBM/Logistic Regression and SMOTE vs. `class_weight` on the full 1,000,000-row dataset with 5-fold cross-validation, and found LightGBM gives a small, real, CV-confirmed edge (full detail in `report/findings.md`). The production `train_churn.py` and `api.py` were updated to match, and re-verified against a live `/predict-churn` request before moving on.
- **Added `libgomp1` to all three Docker images (`api`, `dashboard`, `airflow`).** After switching to LightGBM, the `api` container crashed on startup with `OSError: libgomp.so.1: cannot open shared object file` — LightGBM's native library dlopen()s GNU OpenMP at import time, which the `python:3.11-slim` base image doesn't include by default. The same import chain reaches the dashboard (via `src.model.train_churn`) and the Airflow retrain task, so all three images needed the fix, not just the one that happened to surface it first.
- **Gave the three Airflow services (`airflow-init`, `airflow-webserver`, `airflow-scheduler`) one shared `image:` tag instead of letting each build separately.** Without an explicit `image:` on the shared `x-airflow-common` anchor, Docker Compose tags each service's build by service name even when they share the exact same Dockerfile/context — so rebuilding `airflow-init` alone (to pick up the `libgomp1` fix) silently left `airflow-webserver`/`airflow-scheduler` on the stale pre-fix image. Caught by explicitly testing `import lightgbm` inside the scheduler container after a rebuild that looked successful.
- **Rewrote the dashboard's data layer to stream instead of loading the whole mart.** Once `customer_360` reached 1,000,000 rows the dashboard OOM-killed in its container: it was loading every row into a DataFrame (~620MB) and then letting the preprocessor materialize a dense 1M x 54 float64 matrix (412MB) on top, just to surface ~100 at-risk customers. It now makes one streaming pass that scores in batches and keeps only `(customer_id, churn_probability, monthlycharges)` per customer (~40MB), pulls full rows for the handful actually displayed by id, and computes every segment chart as a SQL `GROUP BY` rather than a pandas groupby over a million rows. Peak memory is now bounded by batch size rather than table size.
- **Discovered `pd.read_sql(chunksize=...)` does not bound memory with psycopg2.** psycopg2's default cursor is client-side: libpq buffers the *entire* result set before pandas sees a row, so `chunksize` only chunks an already-materialized buffer. This was the real cause behind several OOMs that earlier "chunked read" fixes only partially mitigated, and at 1M rows it fails outright with `out of memory for query result`. All warehouse reads now go through `src/warehouse.py`'s `stream_query`, which uses a named (server-side) cursor.
- **Made the recommender serve every customer, not just those in its index.** The k-NN artifact was trained on ~154K customers while `customer_360` held 1M, so id-based lookup left **84.6% of at-risk customers with no recommendation at all** in the dashboard. Added `recommend_for_profile`, which transforms any customer's profile through the saved preprocessor and matches it against the index - reframing the index as a bounded *reference set* rather than a registry of everyone. Coverage went from 15.4% to 100% with no increase in artifact size, and the profile matrix was downcast to float32 (halving it) while there.
- **Raised the dashboard container's `mem_limit` from 400MB to 768MB.** That limit was set when `customer_360` was tiny; as customer_360 grew past ~300K rows, loading the full mart into pandas plus a loaded model plus Streamlit's own baseline footprint got OOM-killed (exit 137) under the old cap.
- **Added probability calibration (`CalibratedClassifierCV`, sigmoid), and it broke the dashboard's SHAP column silently.** `class_weight="balanced"` produced scores 3–5x higher than true probabilities (mean predicted 0.41 vs actual 0.10), which calibration fixes — see [Probability calibration](#probability-calibration). But `CalibratedClassifierCV` has no `.named_steps`, which the dashboard's `column_importances` and SHAP `TreeExplainer` both call directly on the artifact's `pipeline`. The failure mode was the worst kind: `column_importances` already catches `AttributeError` and returns `{}` for a different, legitimate reason (older artifact shapes), so this new failure was swallowed by that same handler and the "Risk factors (SHAP)" column silently rendered empty with no exception anywhere. Caught by reading the actual screenshot after the retrain rather than trusting that "the API responds correctly" meant the dashboard was fine too. Fixed by saving the pre-calibration pipeline in the artifact as `base_pipeline` and routing explainability through it, and pinned with 5 new tests in `tests/test_dashboard.py` that build both pipeline shapes directly and assert the routing rule holds.
- **The two Groq models originally chosen for the AI Agent Layer (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) had been fully retired from Groq's catalog by the time this ran end to end.** The very first live call 404'd with `model_not_found` - not an access issue, confirmed by listing `client.models.list()` and finding neither model present at all. Groq's actual current lineup (`openai/gpt-oss-120b`/`-20b`, `qwen/qwen3.6-27b`, `groq/compound`, …) doesn't include either. Replaced with `openai/gpt-oss-120b`/`-20b`, the nearest equivalent pairing Groq does serve, in the same quality/speed roles originally intended. This is real production risk worth naming plainly: pinning a third-party model ID is pinning something that can be retired out from under you with no warning, unlike a pinned pip package version.
- **Both replacement models are reasoning models, and the first real call came back with an empty completion.** `openai/gpt-oss-*` spend part of the token budget on hidden chain-of-thought before the visible answer - measured, a trivial 3-word reply ("connection ok") cost 50-86 total tokens at default settings, so the original `max_tokens=200-250` budgets were consumed by reasoning before any visible text was produced. Fixed two ways: added `reasoning_effort="low"` (cut the same trivial reply's token cost from 50 to 17 with no visible quality loss on these short, constrained-output tasks) and raised every agent's `max_tokens` to 400-500 to leave headroom for both the reasoning overhead and the actual output.
- **`src/agents/cache.py`'s read path had a bug that would have crashed the very first real call.** `get_or_generate()` checks the cache before generating, but `get_cached_content()` (the read) didn't call `ensure_tables()` the way `put_cached_content()` (the write) did - on a fresh warehouse where nothing had ever been cached yet, that first read hit `relation "public.llm_explanations" does not exist` instead of correctly reporting a cache miss. Caught immediately during live verification (the first real cache test), before it ever reached the dashboard. Fixed by having the read path check `to_regclass()` first and return `None` (a normal miss) when the table doesn't exist yet, matching the pattern `src/monitoring/drift.py` already used for the same class of problem.
- **Real rate limiting was hit and handled, not just simulated in tests.** Generating AI explanations and outreach drafts for 15 real customers in one dashboard session (up to 30 real calls in quick succession) triggered genuine `429 Too Many Requests` responses from Groq's free tier mid-run, visible in the container logs. The exponential-backoff retry in `groq_client.complete()` recovered every one of them - the dashboard session completed with real AI content for all 15 customers, no failures surfaced to the user.
- **Preparing the Streamlit Cloud deployment surfaced two real bugs that a plain `import` of the dashboard module never would have caught** - both only appeared once the app was actually run with `streamlit run` against the trimmed dependency set, not merely imported. (1) The `st.secrets` bridge added for Streamlit Cloud (see "Public demo deployment") was placed before `st.set_page_config()`, and merely *accessing* `st.secrets` counts as a Streamlit command - so `set_page_config()` was no longer literally first, and every page load crashed with `StreamlitSetPageConfigMustBeFirstCommandError`. Fixed by moving `set_page_config()` to the true first line of the script. (2) Even wrapped in `try/except`, *accessing* `st.secrets` at all makes Streamlit render its own "No secrets found" warning banner across the top of every page when no `secrets.toml` exists - which is true for every local and Docker run, not just this test, so without a fix that banner would have appeared on every regular use of the dashboard, not just the Streamlit Cloud deploy. Fixed by checking for the secrets file's existence with a plain path check first, only touching `st.secrets` at all when a real secrets file is present.
- **Also caught while preparing this: `requirements-streamlit-cloud.txt`'s first install attempt silently produced a broken environment** - `pandas`, `streamlit`, `lightgbm`, `shap`, and `groq` were all missing afterward, with the actual cause (`AssertionError` inside pip itself) buried in output that a `-q`/tailed check didn't surface. Root cause: two `pip install` runs were started concurrently into environments whose file operations collided. A clean, single, sequential install - after also upgrading pip itself (22.3.1 had this failure mode; 26.2.1 did not) - succeeded and was verified by actually running the app under `streamlit run`, not just checking `pip list`.
- **`scripts/migrate_to_neon.sh`'s first version assumed a `psql` binary on the host running it - this machine doesn't have one.** Discovered by actually running the script rather than just reading it: `command -v psql` came back empty on this Git Bash / Windows host, which would have failed the restore step outright for anyone in the same position. Fixed by routing both the dump (from local) and the restore (to the target) through the `postgres` container's own `psql`/`pg_dump` instead of a host binary - Docker containers have outbound internet access by default, so the container reaches an external target like Neon exactly as well as it reaches the local `postgres` service. Re-verified by running the fixed script for real against the live Neon database (it's idempotent - `--clean --if-exists` - so re-running it was itself part of the verification).
- **The live Streamlit Cloud deploy failed for real** (`psycopg2-binary==2.9.10` has no wheel for the Python 3.14 environment the platform's build used, so it fell back to a from-source build that needs `pg_config` - absent there, so the build failed outright) **- and the fix was independently re-verified with direct evidence, not just re-asserted.** With Docker unavailable for a second verification pass, `pip download psycopg2-binary==2.9.10 --platform manylinux2014_x86_64 --python-version 3.14 --only-binary=:all:` was run against the real PyPI index and confirmed **zero** matching distribution (pip's own resolver: "Could not find a version that satisfies the requirement... from versions: 2.9.11, 2.9.12, 2.9.13") - the same command against `2.9.12` (the version this repo now pins) downloaded a real `cp314-manylinux2014_x86_64` wheel. The fixed driver was then connected to the live Neon database through the actual production code path (`src/warehouse.py`'s `get_pg_conn()`, not a substitute), returning the correct 1,000,000-row count, and a completely fresh, sequential-only virtualenv install of `requirements-streamlit-cloud.txt` (avoiding the concurrent-install bug documented above) ran the real dashboard end to end against that live database with no crash. A driver migration to `psycopg` (v3) was considered and deliberately rejected: only `psycopg2.connect()` itself is called directly across the 5 files that use it, but `src/warehouse.py`'s named-cursor streaming (the core memory-safety mechanism this whole project depends on) is exactly the kind of subtly-different API surface between the two drivers, and the version bump alone was already proven sufficient - migrating would have been unjustified risk for no remaining benefit. A `runtime.txt` (`python-3.11`) was added for explicitness, but is documented as unreliable rather than relied upon: Streamlit Community Cloud has a live, widely-reported 2026 platform bug where `runtime.txt` and even its own Advanced-Settings Python-version selector are both ignored.

## Limitations

- **The dataset is synthetic, not real telecom customer data.** Every "finding" in `report/findings.md` and the dashboard describes patterns the data generator produced, not an actual market.
- **DuckDB was used in place of Spark due to local hardware constraints (8GB RAM, often <1GB free in practice).** The pipeline's per-batch DuckDB step is fast at this scale (~77K rows/batch, well under a second); it was not tested against a distributed engine at larger scale.
- **Simulated batch ingestion replays a static, pre-existing dataset rather than genuine streaming arrivals**, and batches are triggered manually rather than on a wall-clock schedule (see Deviations).
- **The recommendation system uses content-based filtering only** (k-NN over demographic/account/usage profile vectors, recommending services popular among a customer's nearest neighbors). A real production system would likely combine this with collaborative filtering and online feedback signals.
- **Drift-triggered retraining is implemented, but this dataset cannot exercise it.** The pipeline computes per-feature PSI for every arriving batch and retrains when any feature crosses the 0.25 band (see [Drift monitoring](#drift-monitoring)). Because all 13 batches are slices of one pre-shuffled file, the measured drift is ~0.0004 across every feature and the cadence rule (every 3rd batch) is what actually fires in practice. The detector's ability to fire is demonstrated in `tests/test_drift.py`, not by this data.
- **Churn model performance is modest, and this was investigated, not assumed.** A dedicated experiment (`notebooks/model_dev_offline.py`) compared 4 model families, SMOTE vs. class-weighting, and cross-validated the winner on the full 1M-row dataset. Best result: LightGBM, AUC 0.683 ± 0.002 (5-fold CV) — a small, real improvement over the original RandomForest (0.677), but nothing tried closes the gap to a "strong" classifier. This looks like a genuine ceiling in the synthetic data's individual-level signal (segment-level signal is much stronger — e.g. contract type alone separates a 4.8x churn-rate gap) rather than a fixable modeling gap.
- **`/recommend` answers for any customer in the warehouse, but not identically fast for all of them.** The recommender's k-NN reference set is deliberately bounded (~154K profiles) to keep the artifact small; a customer outside it (about 85% of the 1M, so the normal case) triggers one indexed warehouse lookup and is then matched against that index. Coverage is 100% — verified on a random sample of 40 real customer IDs — but those requests do touch Postgres, unlike `/predict-churn`, which remains purely in-memory.
- **Per-customer explanations are real SHAP values, computed only for the displayed subset.** `TreeExplainer` runs against the bounded set of rows actually on screen (at most 500), never the full 1M-row table — an intentional cost/scale trade-off, not a full-population attribution.
- **The retired Streamlit dashboard's verification included real browser screenshots** (headless Chromium via Playwright against the running container), in addition to Streamlit's `AppTest` runner. Two rendering defects were found and fixed this way that `AppTest` could not surface, because both were layout problems rather than exceptions: a KPI value truncated to `$26,051...`, and the SHAP column clipped mid-phrase.
- **This machine's network was unusually slow throughout the build** (a 40MB Kaggle download took ~11 minutes; the Airflow Docker image took well over an hour to build, twice, due to the sqlalchemy-pin fix requiring a second build). If you rebuild on a faster connection, expect this to go much quicker.
- **The AI Agent Layer is a presentation layer over frozen predictions, not a modeling improvement** - it never sees a feature the model didn't already use, and it cannot change a churn probability, a SHAP ranking, or a recommendation. Two real, specific caveats from live verification: (1) SHAP factor names for categorical features collapse to the base column (e.g. "contract", not "month-to-month") because that mapping is pre-existing and shared with the dashboard's SHAP column, so an explanation can say a factor matters without saying which value of it does; (2) the exact Groq model IDs pinned here (`openai/gpt-oss-120b`/`-20b`) are a live external dependency that already moved once during this project (see Deviations) and could again - unlike a pinned pip package, there's no local copy to fall back to, only the graceful-degradation-to-raw-data behavior this layer was built with from the start.
- **The Neon side of the public demo deployment is now verified against a real, live account** (this session still has no cloud credentials of its own - account creation genuinely needed a human - but once the connection string existed, the migration and the dashboard were both run for real against it, not a substitute). The one number that remains genuinely unverified is first-load latency on the actual Streamlit Community Cloud deployment: this development machine measured ~8 minutes for the full 1,000,000-row query against Neon's `us-east-2` region, but that measurement is from wherever this machine physically is, not from Streamlit Cloud's own (US-based, likely much closer to `us-east-2`) infrastructure - see Verified state and Deviations for the full measurement and why the two numbers likely differ substantially.

## License

[MIT](LICENSE) — the code is free to use, modify, and learn from. The dataset it runs on is a third-party synthetic Kaggle dataset (`isandeep06/customer-churn-prediction-dataset-1m`, see [A note on the data](#a-note-on-the-data)) and is not redistributed here; it is not covered by this license.

## Author

Ly Henglong (Penguin) — Final-year ICT student, American University of Phnom Penh, focused on data science and machine learning.

- LinkedIn: [linkedin.com/in/lyhenglong](https://linkedin.com/in/lyhenglong)
- GitHub: [github.com/LyHenglong](https://github.com/LyHenglong)
