"""Slack webhook alerting - optional, off by default.

Uses urllib (stdlib) rather than adding requests/httpx as a new dependency
for one JSON POST - same reasoning as src/dashboard/app.py's
call_assistant_api(). Every caller must be able to treat a failed/
unconfigured alert as a no-op, never as a reason to fail the pipeline
step that triggered it - the same graceful-degradation convention this
codebase already uses for Groq (src/agents/groq_client.py) and MLflow
(src/model/registry.py): a notification channel being down must never
take down the thing it's supposed to be notifying about.

SLACK_WEBHOOK_URL unset (the default - nothing in .env.example sets a
real value) means send_slack_alert() always returns False without making
any network call. Create an "Incoming Webhook" at
https://api.slack.com/messaging/webhooks and set SLACK_WEBHOOK_URL to
enable it.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger("monitoring.alerting")


def send_slack_alert(text: str, timeout: float = 5.0) -> bool:
    """POSTs `text` to SLACK_WEBHOOK_URL if set. Returns whether it was
    actually sent - never raises, so a Slack/network problem can't fail
    whatever pipeline step is reporting the alert."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return False

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                log.warning("Slack webhook returned status %s", resp.status)
            return ok
    except Exception as exc:
        log.warning("Slack alert failed to send (continuing regardless): %s", exc)
        return False
