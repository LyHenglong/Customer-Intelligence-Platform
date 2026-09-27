# Common development tasks. `make help` lists them.

PYTHON ?= python
COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test test-integration coverage eval-retrieval compare-models \
	frontend-install frontend-check frontend-test up serve down dbt-build api

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the full Python runtime (API, training, RAG)
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install the lighter CI/test toolchain and pre-commit hooks
	$(PYTHON) -m pip install -r requirements-ci.txt pre-commit
	pre-commit install

lint: ## Lint and check formatting (Python)
	ruff check .
	ruff format --check .

format: ## Auto-fix lint findings and format Python code
	ruff check --fix .
	ruff format .

test: ## Unit tests (no database or network needed)
	$(PYTHON) -m pytest

coverage: ## Unit tests with a coverage report
	$(PYTHON) -m pytest --cov --cov-report=term --cov-report=html

test-integration: ## Integration tests against a warehouse built as in CI (see .github/workflows/ci.yml)
	RUN_INTEGRATION=1 $(PYTHON) -m pytest tests/integration -v

eval-retrieval: ## Offline BM25 retrieval quality over knowledge/
	$(PYTHON) -m src.ai.evaluation.offline_retrieval --k 1

compare-models: ## Benchmark Groq models on the real agent prompts (needs GROQ_API_KEY; ~30 cheap calls)
	$(PYTHON) -m scripts.compare_groq_models --out report/groq_model_comparison.md

frontend-install: ## Install frontend dependencies
	cd frontend && npm ci

frontend-check: ## Frontend lint, typecheck and production build
	cd frontend && npm run lint && npm run typecheck && npm run build

frontend-test: ## Frontend Playwright smoke tests (run frontend-check first)
	cd frontend && npm test

up: ## Start the warehouse and Airflow
	$(COMPOSE) up -d postgres airflow-init airflow-webserver airflow-scheduler

serve: ## Start the API and the frontend
	$(COMPOSE) --profile serving up -d api frontend

down: ## Stop everything
	$(COMPOSE) --profile serving --profile production down

dbt-build: ## Run dbt models and data tests against the configured warehouse
	cd dbt && dbt build --profiles-dir .

api: ## Run the API locally with auto-reload
	uvicorn src.api.app:app --reload
