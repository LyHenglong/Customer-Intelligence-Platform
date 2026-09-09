# Telecom Customer Churn & Recommendation Platform

A production-style data platform that predicts telecom customer churn and recommends services to at-risk customers, built on a large-scale synthetic dataset to demonstrate a realistic enterprise data pipeline: simulated batch ingestion, distributed-style processing, a transformation layer, a unified customer view, predictive modeling, and live serving — all orchestrated end to end.

## Motivation

Customer churn is one of the most common and well-understood problems in the data science field, and pairing it with a recommendation system demonstrates two distinct modeling techniques within one coherent business narrative: **predict who's about to leave, then recommend something that gives them a reason to stay.**

Rather than analyze a small, pre-cleaned dataset in a notebook, this project is built as a small but complete enterprise-style data platform — using tools and patterns found in real production data teams — to make the underlying engineering, not just the modeling, part of the deliverable.

## A note on the data

This project uses a **large-scale synthetic** telecom churn dataset (~1 million customer records, generated to mimic real-world patterns). It is **not real customer data**, and any patterns or findings described in this project reflect the synthetic dataset's generated structure, not an actual telecom market. The dataset was chosen specifically for its size — large enough to meaningfully exercise tools built for scale (DuckDB, dbt, Airflow) rather than a small dataset that pandas alone could handle just as easily.

## Why DuckDB instead of Spark

The natural choice for "big data" processing is often Apache Spark. This project deliberately uses **DuckDB** instead — an in-process analytical database that handles millions of rows efficiently with a fraction of Spark's memory footprint, and has a native dbt adapter. On a resource-constrained development machine (8GB RAM), running Spark alongside Airflow, Postgres, and the rest of the stack simultaneously is unreliable. DuckDB is increasingly used in real companies for exactly this reason: it delivers most of the analytical performance benefit of a distributed engine without the operational overhead, for workloads that don't genuinely require a multi-node cluster. This is a deliberate engineering trade-off, not a workaround — 1M rows is well within DuckDB's comfortable range and does not require distributed computing.

## Project Goals

- Simulate a realistic batch-ingestion pattern (weekly-style arrivals) from a static dataset
- Process data at meaningful scale using DuckDB rather than pandas alone
- Build a transformation layer with dbt, including a unified **Customer 360** view
- Train and evaluate a churn prediction model (classification)
- Build a content-based recommendation system for service upsell/retention offers
- Serve both models live via a FastAPI service
- Present findings and at-risk customers through an interactive Streamlit dashboard
- Orchestrate the full pipeline with Airflow, including a periodic retraining step
- Apply software engineering practices throughout: testing, CI/CD, containerization

## Architecture

```
1M-row synthetic dataset (simulated weekly batches)
        │
        ▼
   Airflow DAG triggers ingestion on schedule
        │
        ▼
   DuckDB processing  ──────────►  cleaning, type validation,
        │                          feature engineering at scale
        ▼
   Raw tables in PostgreSQL (warehouse)
        │
        ▼
      dbt run  ────────────────►  staging models → intermediate models
        │                          → customer_360 mart (tested, documented)
        ▼
  customer_360 (one row per customer, everything consolidated)
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
  Churn model (train.py)      Recommendation model (train.py)   Streamlit dashboard
        │                              │                          (reads customer_360
        ▼                              ▼                           directly)
  FastAPI /predict-churn      FastAPI /recommend
        │                              │
        └──────────────┬───────────────┘
                        ▼
              Periodic retraining
        (triggered after N new batches land,
         versioned model artifacts, not overwritten)
```

## Repository Structure

```
.
├── dags/
│   └── churn_pipeline.py         # Airflow DAG: ingest -> DuckDB -> load -> dbt run/test -> retrain check
├── src/
│   ├── ingest/
│   │   └── batch_loader.py       # picks up next simulated batch, loads via DuckDB, writes to Postgres
│   ├── model/
│   │   ├── train_churn.py        # trains and evaluates the churn classifier
│   │   ├── train_recommender.py  # trains the content-based recommendation model
│   │   └── api.py                # FastAPI app: /predict-churn and /recommend
│   └── dashboard/
│       └── app.py                # Streamlit dashboard
├── dbt/
│   ├── models/
│   │   ├── staging/               # cleaned, typed versions of raw tables
│   │   ├── intermediate/          # joins, aggregations (usage summaries, subscription flags)
│   │   └── marts/
│   │       └── customer_360.sql   # unified customer view
│   ├── tests/                     # dbt tests: not_null, unique, accepted_values
│   └── dbt_project.yml
├── db/
│   └── schema.sql                 # raw table definitions
├── data/
│   └── raw/                       # simulated batch files (batch_001.csv ... batch_0NN.csv)
├── notebooks/
│   ├── eda.ipynb
│   └── model_dev.ipynb
├── tests/
│   └── test_ingest.py             # unit tests for ingestion/cleaning logic
├── report/
│   └── findings.md                # business-facing write-up (with synthetic-data caveat)
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── docker-compose.yml             # Postgres, Airflow, API, dashboard
├── .github/workflows/ci.yml       # test-on-push CI pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## The Customer 360 concept

**Customer 360** is a standard enterprise data pattern: a single, unified table consolidating everything known about a customer — demographics, account details, service subscriptions, usage, billing, churn risk, and recommended actions — instead of that information being scattered across tables that must be joined every time it's needed.

In this project, `customer_360` is the final dbt mart model: one row per customer, built by joining the staging and intermediate models. It is the single source of truth that both the churn model, the recommendation model, and the Streamlit dashboard read from — mirroring how real Customer Data Platforms (CDPs) and CRM systems are structured.

## Methodology

### 1. Simulated ingestion
The static dataset is split into ~10-15 batch files to simulate weekly production arrivals. Each Airflow DAG run picks up the next unprocessed batch — a standard technique (replaying historical data through a pipeline) used even in real engineering teams to test pipelines before they go live with genuine streaming data.

### 2. Processing (DuckDB)
Each batch is loaded and processed with DuckDB: type validation, cleaning, and feature engineering at a scale meaningful enough to justify the tool choice, before being written into the PostgreSQL warehouse.

### 3. Transformation (dbt)
dbt builds staging models (typed, cleaned tables), intermediate models (joins and aggregations), and the `customer_360` mart, with tests enforcing data quality (not-null, uniqueness, accepted value ranges) at each layer.

### 4. Predictive modeling
- **Churn model**: a classifier trained on `customer_360`, evaluated with precision, recall, and F1 score rather than raw accuracy, since churn datasets are typically imbalanced.
- **Recommendation model**: content-based filtering that recommends un-subscribed services to a customer based on similarity to other customers' profiles and subscriptions.

### 5. Serving
Both trained models are served live via FastAPI endpoints (`/predict-churn`, `/recommend`) rather than existing only inside notebooks.

### 6. Retraining
After a defined number of new simulated batches have landed, the Airflow DAG triggers retraining on the updated `customer_360` table, saving a new versioned model artifact rather than silently overwriting the previous one — demonstrating the full production ML lifecycle loop (train → serve → monitor → retrain).

### 7. Dashboard
The Streamlit dashboard presents at-risk customers (churn probability, key risk factors) alongside recommended retention actions, plus aggregate charts on churn patterns across contract types, tenure, and service bundles.

## Key Findings

*(To be filled in once analysis is complete. Remember: this dataset is synthetic — phrase findings as patterns observed in the generated data, not real-world market claims.)*

## Limitations

- The dataset is synthetic, not real telecom customer data; findings describe patterns in generated data, not an actual market
- DuckDB was used in place of Spark due to local hardware constraints (8GB RAM); the pipeline is designed to scale conceptually to a distributed engine but was not tested at that scale
- Simulated batch ingestion replays a static, pre-existing dataset rather than genuine streaming arrivals
- The recommendation system uses content-based filtering only; a real production system would likely combine this with collaborative filtering and online feedback signals
- Retraining is triggered on a simple batch-count rule rather than genuine model-drift monitoring, which a real production system would use instead

## How to Run

### Local development (Docker Compose)

```bash
# spin up Postgres, Airflow, API, and dashboard
docker-compose up -d

# access Airflow UI to trigger/monitor the DAG
open http://localhost:8080

# access the dashboard
open http://localhost:8501

# access the API docs
open http://localhost:8000/docs
```

### dbt

```bash
cd dbt
dbt run
dbt test
```

### Running tests

```bash
pytest tests/
```

### Notebooks (for exploration/development)

```bash
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/model_dev.ipynb
```

## Tech Stack

| Category | Tools |
|---|---|
| Languages | Python, SQL |
| Orchestration | Apache Airflow |
| Processing | DuckDB |
| Transformation | dbt |
| Storage | PostgreSQL |
| Modeling | scikit-learn, pandas |
| Model Serving | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest, dbt tests |

## Author

Ly Henglong (Penguin) — Final-year ICT student, American University of Phnom Penh, focused on data science and machine learning.

- LinkedIn: [linkedin.com/in/lyhenglong](https://linkedin.com/in/lyhenglong)
- GitHub: [github.com/LyHenglong](https://github.com/LyHenglong)
