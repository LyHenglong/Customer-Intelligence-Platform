"""FastAPI serving layer: churn/recommender predictions, the AI Agent Layer
that narrates them, the dashboard-facing analytics endpoints the Next.js
frontend reads, and the AI decision assistant.

    app.py      - app assembly, CORS, startup lifecycle (model load + cache warm)
    state.py    - the in-memory model artifacts every router serves from
    schemas.py  - request/response models
    security.py - API-key auth and rate limiting for the costly endpoints
    routers/    - one module per endpoint group
"""
