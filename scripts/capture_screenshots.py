"""
Captures the dashboard screenshots used in README.md.

Run against a dashboard that is already serving:

    docker compose --profile serving up -d dashboard
    pip install playwright && playwright install chromium
    python scripts/capture_screenshots.py

Images are written to docs/images/ and overwrite the committed ones, so a
re-run after a UI change refreshes the README rather than accumulating
files.

Two things here are less obvious than they look, and both caused wrong
captures before they were handled:

1. Streamlit streams the page in. The DOM (and even the tab bar) exists
   long before the content does, so a naive `wait_for_selector` screenshots
   a spinner and an empty table. `wait_idle` polls the status widget's text
   and requires several consecutive idle reads before shooting.
2. `full_page=True` does not work here. Streamlit scrolls an inner
   container rather than the document, so Playwright's full-page capture
   returns just the viewport. Getting a tall section into one image means
   using a taller viewport, which is what DRIFT_VIEWPORT is for.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - developer tooling, not app code
    sys.exit("playwright is not installed. Run: pip install playwright && playwright install chromium")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "images"

DEFAULT_URL = "http://localhost:8501"
VIEWPORT = {"width": 1600, "height": 1000}
# Taller, so the Feature drift section fits in a single image (see note 2).
DRIFT_VIEWPORT = {"width": 1600, "height": 1700}

# view name -> a selector that only exists once that view's real content
# has rendered, so we never capture a half-built page.
VIEWS = [
    ("Overview", '[data-testid="stPlotlyChart"]'),
    ("At-Risk Customers", '[data-testid="stDataFrame"]'),
    ("Segments", '[data-testid="stPlotlyChart"]'),
    ("Model Performance", '[data-testid="stMetric"]'),
    ("Pipeline Status", '[data-testid="stDataFrame"]'),
]

# Scoring 1M customers on a cold cache genuinely takes minutes.
LOAD_TIMEOUT_MS = 420_000
IDLE_TIMEOUT_S = 420


def wait_idle(page, timeout_s: int = IDLE_TIMEOUT_S) -> None:
    """Blocks until Streamlit's status widget stops reporting RUNNING."""
    deadline = time.time() + timeout_s
    consecutive_idle = 0
    while time.time() < deadline:
        try:
            widget = page.locator('[data-testid="stStatusWidget"]')
            busy = widget.count() > 0 and "RUNNING" in (widget.first.inner_text() or "").upper()
        except Exception:
            # The widget unmounts between reruns; a missing widget is idle.
            busy = False
        consecutive_idle = 0 if busy else consecutive_idle + 1
        if consecutive_idle >= 3:
            break
        time.sleep(2)
    page.wait_for_timeout(2500)  # let fonts and plotly finish painting


def _open(browser, url: str, viewport: dict):
    page = browser.new_page(viewport=viewport, device_scale_factor=2)
    page.set_default_timeout(LOAD_TIMEOUT_MS)
    page.goto(url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)
    page.wait_for_selector('[data-testid="stMetric"]', timeout=LOAD_TIMEOUT_MS)
    wait_idle(page)
    return page


def capture_views(browser, url: str) -> int:
    failures = 0
    page = _open(browser, url, VIEWPORT)
    for name, content_selector in VIEWS:
        try:
            if name != "Overview":
                page.get_by_text(name, exact=True).first.click()
                wait_idle(page)
            page.wait_for_selector(content_selector, timeout=LOAD_TIMEOUT_MS)
            wait_idle(page)

            path = OUT_DIR / f"dashboard-{name.lower().replace(' ', '-')}.png"
            page.screenshot(path=str(path))
            print(f"  OK   {path.name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    page.close()
    return failures


def capture_drift(browser, url: str) -> int:
    """The drift section sits below the fold, so it gets its own pass."""
    page = _open(browser, url, DRIFT_VIEWPORT)
    try:
        page.get_by_text("Pipeline Status", exact=True).first.click()
        wait_idle(page)
        page.get_by_text("Feature drift (PSI)", exact=True).first.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
        path = OUT_DIR / "dashboard-drift.png"
        page.screenshot(path=str(path))
        print(f"  OK   {path.name}")
        return 0
    except Exception as exc:
        print(f"  FAIL drift: {type(exc).__name__}: {exc}")
        return 1
    finally:
        page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"dashboard URL (default: {DEFAULT_URL})")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capturing {args.url} -> {OUT_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            failures = capture_views(browser, args.url) + capture_drift(browser, args.url)
        finally:
            browser.close()

    if failures:
        print(f"{failures} capture(s) failed")
        return 1
    print("All captures succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
