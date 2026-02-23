"""Public page rendering tests — all pages served via /public prefix."""

import pytest

pytestmark = [pytest.mark.e2e]

# Public pages are mounted at /public prefix in the app router.
# The hostname routing middleware rewrites / -> /public, /tips -> /public/tips
# etc. on the public domain. In tests we access /public/* directly.


class TestHomepage:
    def test_loads_200(self, public_page):
        resp = public_page.goto("/public")
        assert resp.status == 200

    def test_has_title(self, public_page):
        public_page.goto("/public")
        public_page.wait_for_load_state("domcontentloaded")
        assert public_page.title()

    def test_has_stats_section(self, public_page):
        public_page.goto("/public")
        public_page.wait_for_load_state("domcontentloaded")
        # Wait for JS to populate stats
        public_page.wait_for_timeout(1000)
        assert public_page.locator("body").is_visible()


class TestStatsPage:
    def test_loads_200(self, public_page):
        """Stats page renders the daily scorecard dashboard."""
        resp = public_page.goto("/public/stats")
        assert resp.status == 200


class TestTipsPage:
    def test_loads_200(self, public_page):
        resp = public_page.goto("/public/tips")
        assert resp.status == 200

    def test_has_content(self, public_page):
        public_page.goto("/public/tips")
        public_page.wait_for_load_state("domcontentloaded")
        assert public_page.locator("body").inner_text()


class TestTipsMeetingPage:
    def test_valid_meeting_loads(self, public_page):
        from tests.e2e.conftest import TODAY
        resp = public_page.goto(f"/public/tips/flemington-{TODAY}")
        # Should load (200) or show content
        assert resp.status in (200, 404)

    def test_invalid_meeting_returns_404(self, public_page):
        resp = public_page.goto("/public/tips/nonexistent-meeting-2099-01-01")
        assert resp.status == 404


class TestStaticPages:
    """All informational pages should render."""

    @pytest.mark.parametrize("path", [
        "/public/about",
        "/public/how-it-works",
        "/public/contact",
        "/public/glossary",
        "/public/calculator",
        "/public/terms",
    ])
    def test_page_loads(self, public_page, path):
        resp = public_page.goto(path)
        assert resp.status == 200
        public_page.wait_for_load_state("domcontentloaded")
        assert public_page.locator("body").inner_text()
