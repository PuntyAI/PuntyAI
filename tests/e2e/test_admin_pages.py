"""Admin page rendering tests — requires authenticated session."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestDashboard:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/")
        assert resp.status == 200

    def test_has_content(self, auth_page):
        auth_page.goto("/")
        auth_page.wait_for_load_state("domcontentloaded")
        body_text = auth_page.locator("body").inner_text()
        assert body_text  # Not empty


class TestMeetsPage:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/meets")
        assert resp.status == 200

    def test_has_content(self, auth_page):
        auth_page.goto("/meets")
        auth_page.wait_for_load_state("domcontentloaded")
        assert auth_page.locator("body").inner_text()


class TestMeetDetailPage:
    def test_valid_meeting_loads(self, auth_page):
        from tests.e2e.conftest import TODAY
        resp = auth_page.goto(f"/meets/flemington-{TODAY}")
        assert resp.status == 200

    def test_shows_races(self, auth_page):
        from tests.e2e.conftest import TODAY
        auth_page.goto(f"/meets/flemington-{TODAY}")
        auth_page.wait_for_load_state("domcontentloaded")
        assert auth_page.locator("body").inner_text()


class TestContentPage:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/content")
        assert resp.status == 200


class TestReviewPage:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/review")
        assert resp.status == 200

    def test_review_detail_loads(self, auth_page):
        resp = auth_page.goto("/review/e2e-content-rand")
        # Content with pending_review should be accessible
        assert resp.status == 200


class TestSettingsPage:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/settings")
        assert resp.status == 200

    def test_has_content(self, auth_page):
        auth_page.goto("/settings")
        auth_page.wait_for_load_state("domcontentloaded")
        assert auth_page.locator("body").inner_text()


class TestLearningsPage:
    def test_loads_200(self, auth_page):
        resp = auth_page.goto("/settings/learnings")
        assert resp.status == 200
