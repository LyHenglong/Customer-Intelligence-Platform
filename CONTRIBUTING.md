# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11
make install-dev        # test toolchain + pre-commit hooks
make frontend-install   # only if you touch frontend/
```

`requirements-ci.txt` is enough for every test. The full `requirements.txt`
adds PyTorch (for the RAG models) and is only needed to run the API with
retrieval or to train models.

## Before you push

| Change touches | Run |
| --- | --- |
| Python | `make lint test` |
| `frontend/` | `make frontend-check frontend-test` |
| dbt models, `db/schema.sql`, ingestion | the `integration` job in `.github/workflows/ci.yml`, or `make test-integration` against a warehouse built the same way |
| `knowledge/`, chunking, BM25 | `make eval-retrieval`; the floors are in `tests/test_ai_evaluation_offline_retrieval.py` |

CI runs all of these on every push: lint, unit tests with a coverage floor,
a Postgres + dbt integration job, and the frontend's lint, typecheck,
build and Playwright smoke tests.

## Conventions

- **Formatting and lint:** ruff, configured in `pyproject.toml`. The
  pre-commit hook runs it for you.
- **Tests never need live infrastructure.** Unit tests mock the database,
  MLflow and Groq. Anything that needs a real Postgres goes in
  `tests/integration/`, which is skipped unless `RUN_INTEGRATION=1`.
- **Numbers in the README and `report/` are measured, never estimated.**
  If a change moves one, re-run what produced it and update the text with
  the new value.
- **API endpoints** live in `src/api/routers/`, one module per area.
  Anything that can call the LLM takes the `limit_llm_calls` dependency.
- **The AI SQL tool's allowlist** is defined in `src/ai/config.py` and
  mirrored by the GRANTs in `db/ai_readonly_role.sql`; change both
  together.
- **Commit messages** explain why, not just what.
