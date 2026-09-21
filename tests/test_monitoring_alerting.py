"""
Tests for src/monitoring/alerting.py's Slack webhook integration.

No real network call in any test - urlopen is monkeypatched. The
important behavior here is that a missing/failed webhook is always a
silent no-op (returns False, never raises), since callers (drift
detection, the Airflow DAG's failure callback) must never have an alert
failure take down the actual pipeline step reporting it.
"""

from __future__ import annotations

import json

import pytest

from src.monitoring import alerting


def test_no_webhook_configured_is_a_noop(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("urlopen must not be called when SLACK_WEBHOOK_URL is unset")

    monkeypatch.setattr(alerting.urllib.request, "urlopen", _should_not_be_called)

    assert alerting.send_slack_alert("test message") is False


def test_successful_post_returns_true(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/fake")
    sent = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr(alerting.urllib.request, "urlopen", _fake_urlopen)

    result = alerting.send_slack_alert("drift detected")

    assert result is True
    assert sent["url"] == "https://hooks.slack.test/fake"
    assert sent["body"] == {"text": "drift detected"}


def test_non_2xx_response_returns_false(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/fake")

    class _FakeResponse:
        status = 404

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(alerting.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse())

    assert alerting.send_slack_alert("drift detected") is False


def test_network_exception_is_caught_not_raised(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/fake")

    def _raise(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(alerting.urllib.request, "urlopen", _raise)

    assert alerting.send_slack_alert("drift detected") is False
