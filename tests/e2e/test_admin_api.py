"""Admin API endpoint tests — requires authenticated session."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestMeetsAPI:
    def test_list_meetings(self, auth_page):
        resp = auth_page.request.get("/api/meets/")
        assert resp.status == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 2  # Flemington + Randwick

    def test_today_meetings(self, auth_page):
        resp = auth_page.request.get("/api/meets/today")
        assert resp.status == 200
        body = resp.json()
        assert isinstance(body, list)


class TestContentAPI:
    def test_list_content(self, auth_page):
        resp = auth_page.request.get("/api/content/")
        assert resp.status == 200
        body = resp.json()
        assert isinstance(body, list)

    def test_review_queue(self, auth_page):
        resp = auth_page.request.get("/api/content/review-queue")
        assert resp.status == 200
        body = resp.json()
        assert isinstance(body, list)
        # Should have at least the Randwick pending_review content
        assert len(body) >= 1

    def test_review_count(self, auth_page):
        resp = auth_page.request.get("/api/content/review-count")
        assert resp.status == 200
        # Returns plain text count for HTMX badge
        text = resp.text()
        assert text.strip().isdigit() or text.strip() == ""


class TestResultsAPI:
    def test_monitor_status(self, auth_page):
        resp = auth_page.request.get("/api/results/monitor-status")
        assert resp.status == 200
        body = resp.json()
        assert "running" in body

    def test_performance(self, auth_page):
        resp = auth_page.request.get("/api/results/performance")
        assert resp.status == 200


class TestSchedulerAPI:
    def test_status(self, auth_page):
        resp = auth_page.request.get("/api/scheduler/status")
        assert resp.status == 200
        body = resp.json()
        assert "running" in body


class TestSettingsAPI:
    def test_list_settings(self, auth_page):
        resp = auth_page.request.get("/api/settings/")
        assert resp.status == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert "unit_value" in body

    def test_weights(self, auth_page):
        resp = auth_page.request.get("/api/settings/weights")
        assert resp.status == 200
        body = resp.json()
        assert "weights" in body
