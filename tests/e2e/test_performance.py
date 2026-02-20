"""Performance tests — page load and API response time assertions."""

import time

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class TestAPIPerformance:
    def test_health_under_200ms(self, page):
        start = time.monotonic()
        resp = page.request.get("/health")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status == 200
        assert elapsed_ms < 200, f"Health took {elapsed_ms:.0f}ms"

    def test_public_stats_under_1s(self, page):
        start = time.monotonic()
        resp = page.request.get("/api/public/stats")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status == 200
        assert elapsed_ms < 1000, f"Stats took {elapsed_ms:.0f}ms"

    def test_bet_type_stats_under_2s(self, page):
        start = time.monotonic()
        resp = page.request.get("/api/public/bet-type-stats")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status == 200
        assert elapsed_ms < 2000, f"Bet type stats took {elapsed_ms:.0f}ms"

    def test_filter_options_under_1s(self, page):
        start = time.monotonic()
        resp = page.request.get("/api/public/filter-options")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status == 200
        assert elapsed_ms < 1000, f"Filter options took {elapsed_ms:.0f}ms"


class TestPageLoadPerformance:
    def test_homepage_domcontentloaded_under_3s(self, public_page):
        start = time.monotonic()
        public_page.goto("/public", wait_until="domcontentloaded")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 3000, f"Homepage DOMContentLoaded took {elapsed_ms:.0f}ms"

    def test_admin_dashboard_under_3s(self, auth_page):
        start = time.monotonic()
        auth_page.goto("/", wait_until="domcontentloaded")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 3000, f"Admin dashboard took {elapsed_ms:.0f}ms"

    def test_tips_page_under_3s(self, public_page):
        start = time.monotonic()
        public_page.goto("/public/tips", wait_until="domcontentloaded")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 3000, f"Tips page took {elapsed_ms:.0f}ms"


class TestConcurrentRequests:
    def test_concurrent_api_calls(self, page):
        """Multiple concurrent API requests should complete quickly."""
        import concurrent.futures
        import httpx
        from tests.e2e.conftest import BASE_URL

        urls = [
            f"{BASE_URL}/health",
            f"{BASE_URL}/api/public/stats",
            f"{BASE_URL}/api/public/wins",
            f"{BASE_URL}/api/public/next-race",
            f"{BASE_URL}/api/public/venues",
        ]

        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(httpx.get, url, timeout=5) for url in urls]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed_ms = (time.monotonic() - start) * 1000

        for r in results:
            assert r.status_code == 200
        assert elapsed_ms < 3000, f"5 concurrent requests took {elapsed_ms:.0f}ms"
