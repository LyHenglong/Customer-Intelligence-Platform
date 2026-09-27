"""HTTP-level observability for the serving API: a request ID on every
response, Prometheus metrics at /metrics, and one structured access-log
line per request (JSON when LOG_FORMAT=json, for log aggregators).

The AI assistant's own per-request traces (tool calls, tokens, cost) live
separately in src/ai/observability/ - this module covers every route.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

log = logging.getLogger("api.access")

REQUESTS = Counter(
    "api_requests_total",
    "HTTP requests handled, by route template, method and status code.",
    ["route", "method", "status"],
)
LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency by route template.",
    ["route", "method"],
    # The dashboard endpoints range from milliseconds (cached) to ~100s (a
    # cold full-population scoring pass), so the buckets span both.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

_REQUEST_ID_HEADER = "X-Request-ID"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    if os.environ.get("LOG_FORMAT", "text").lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _route_template(request: Request) -> str:
    """The matched route's path template (/customers/{customer_id}), never
    the raw path - raw customer IDs as label values would create one time
    series per customer."""
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _observe(request: Request, call_next):
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[_REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = _route_template(request)
            if route != "/metrics":
                REQUESTS.labels(route, request.method, str(status)).inc()
                LATENCY.labels(route, request.method).observe(elapsed)
                log.info(
                    "%s %s %s %.1fms",
                    request.method,
                    request.url.path,
                    status,
                    elapsed * 1000,
                    extra={
                        "fields": {
                            "request_id": request_id,
                            "method": request.method,
                            "path": request.url.path,
                            "route": route,
                            "status": status,
                            "duration_ms": round(elapsed * 1000, 1),
                        }
                    },
                )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
