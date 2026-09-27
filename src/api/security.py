"""API-key auth and rate limiting.

API key: when API_KEYS (comma-separated) is set, every route except /health
requires a matching X-API-Key header. Unset means auth is off - the local
docker-compose default. The Next.js frontend never ships the key to the
browser: its server-side proxy route (frontend/app/api/backend/) adds the
header, so a deployment can set API_KEYS without exposing it.

Rate limiting: the endpoints that can call the LLM (/assistant/query,
/explain-churn, /outreach-draft) are limited per client, so an anonymous
loop cannot burn through the Groq quota. The window is in-process - with
several replicas each enforces its own limit, which bounds total spend at
replicas x limit rather than exactly at the limit.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_keys() -> list[str]:
    return [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]


def auth_enabled() -> bool:
    return bool(_configured_keys())


async def require_api_key(request: Request) -> None:
    keys = _configured_keys()
    if not keys or request.url.path == "/health":
        return
    supplied = await _api_key_header(request)
    # compare_digest per key: a plain `in` check leaks how much of a key
    # matched through response timing.
    if not supplied or not any(hmac.compare_digest(supplied, k) for k in keys):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def client_id(request: Request) -> str:
    """The caller's address. Behind a trusted reverse proxy (Render, Caddy,
    the frontend's own proxy route) every request arrives from the proxy, so
    TRUST_PROXY_HEADERS=true switches to the first X-Forwarded-For hop -
    only safe when something in front of the API sets that header itself."""
    if os.environ.get("TRUST_PROXY_HEADERS", "false").lower() == "true":
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding-window limit of `limit` calls per `window_s` seconds per client."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit = limit
        self.window_s = window_s
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Records a call. Returns None if allowed, otherwise the seconds
        until the oldest call in the window expires."""
        now = time.monotonic()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self.window_s:
                calls.popleft()
            if len(calls) >= self.limit:
                return self.window_s - (now - calls[0])
            calls.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


llm_limiter = RateLimiter(limit=int(os.environ.get("LLM_RATE_LIMIT_PER_MINUTE", "10")))


def limit_llm_calls(request: Request) -> None:
    retry_after = llm_limiter.hit(client_id(request))
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded for AI endpoints, try again shortly",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )
