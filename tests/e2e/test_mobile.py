"""Responsive/mobile viewport tests."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestMobilePublicPages:
    def test_homepage_no_horizontal_overflow(self, mobile_public_page):
        mobile_public_page.goto("/public")
        mobile_public_page.wait_for_load_state("domcontentloaded")
        # Check that page width doesn't exceed viewport
        body_width = mobile_public_page.evaluate("document.body.scrollWidth")
        viewport_width = mobile_public_page.evaluate("window.innerWidth")
        assert body_width <= viewport_width + 5  # 5px tolerance

    def test_stats_page_loads(self, mobile_public_page):
        resp = mobile_public_page.goto("/public/stats")
        assert resp.status == 200

    def test_tips_page_renders(self, mobile_public_page):
        resp = mobile_public_page.goto("/public/tips")
        assert resp.status == 200


class TestMobileAdminPages:
    def test_dashboard_renders(self, mobile_page):
        resp = mobile_page.goto("/")
        assert resp.status == 200
        mobile_page.wait_for_load_state("domcontentloaded")
        body_width = mobile_page.evaluate("document.body.scrollWidth")
        viewport_width = mobile_page.evaluate("window.innerWidth")
        assert body_width <= viewport_width + 5

    def test_meets_page_renders(self, mobile_page):
        resp = mobile_page.goto("/meets")
        assert resp.status == 200


class TestTabletPages:
    def test_dashboard_renders(self, tablet_page):
        resp = tablet_page.goto("/")
        assert resp.status == 200

    def test_public_stats_loads(self, tablet_page):
        resp = tablet_page.goto("/public/stats")
        assert resp.status == 200


class TestDesktopPages:
    def test_dashboard_renders(self, desktop_page):
        resp = desktop_page.goto("/")
        assert resp.status == 200

    def test_public_homepage_renders(self, desktop_page):
        resp = desktop_page.goto("/public")
        assert resp.status == 200
