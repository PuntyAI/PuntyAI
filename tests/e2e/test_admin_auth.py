"""Authentication and CSRF tests."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestUnauthenticatedAccess:
    def test_admin_page_redirects_to_login(self, page):
        """Unauthenticated GET / on non-public host redirects to /login."""
        resp = page.goto("/")
        # Should redirect to login page
        assert "/login" in page.url

    def test_admin_api_returns_401(self, page):
        """Unauthenticated API call returns 401."""
        resp = page.request.get("/api/meets/")
        assert resp.status == 401
        assert resp.json()["detail"] == "Not authenticated"

    def test_meets_page_redirects(self, page):
        page.goto("/meets")
        assert "/login" in page.url

    def test_settings_page_redirects(self, page):
        page.goto("/settings")
        assert "/login" in page.url

    def test_public_paths_always_accessible(self, page):
        """Health and public API are accessible without auth."""
        assert page.request.get("/health").status == 200
        assert page.request.get("/api/public/stats").status == 200

    def test_login_page_renders(self, page):
        page.goto("/login")
        page.wait_for_load_state("domcontentloaded")
        assert page.locator("body").inner_text()


class TestAuthenticatedAccess:
    def test_admin_dashboard_loads(self, auth_page):
        auth_page.goto("/")
        auth_page.wait_for_load_state("domcontentloaded")
        assert "/login" not in auth_page.url

    def test_admin_api_works(self, auth_page):
        resp = auth_page.request.get("/api/meets/")
        assert resp.status == 200

    def test_logout_clears_session(self, auth_page):
        auth_page.goto("/logout")
        # After logout, should be at login page
        assert "/login" in auth_page.url
        # API should now return 401
        resp = auth_page.request.get("/api/meets/")
        assert resp.status == 401


class TestCSRF:
    def test_csrf_token_endpoint(self, auth_page):
        resp = auth_page.request.get("/api/csrf-token")
        assert resp.status == 200
        token = resp.json()["csrf_token"]
        assert isinstance(token, str)
        assert "." in token  # format: nonce.signature

    def test_post_without_csrf_returns_403(self, auth_page):
        """POST without CSRF token should be rejected."""
        resp = auth_page.request.post("/api/meets/scrape-calendar")
        assert resp.status == 403

    def test_post_with_invalid_csrf_returns_403(self, auth_page):
        resp = auth_page.request.post(
            "/api/meets/scrape-calendar",
            headers={"X-CSRF-Token": "invalid.token"},
        )
        assert resp.status == 403

    def test_post_with_valid_csrf_accepted(self, auth_page):
        """POST with valid CSRF should not return 403."""
        from tests.e2e.conftest import get_csrf_token
        token = get_csrf_token(auth_page)
        resp = auth_page.request.post(
            "/api/meets/scrape-calendar",
            headers={"X-CSRF-Token": token},
        )
        # Should not be 403 (CSRF ok). Might be 200 or 500 depending on
        # whether the scraper can reach external APIs in test env.
        assert resp.status != 403
