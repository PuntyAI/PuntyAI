"""Smoke tests — verify E2E infrastructure works."""

import pytest

pytestmark = [pytest.mark.e2e]


def test_health_endpoint(page):
    """Server is running and /health returns 200."""
    resp = page.request.get("/health")
    assert resp.status == 200
    body = resp.json()
    assert body["status"] == "healthy"


def test_public_api_stats(page):
    """/api/public/stats is accessible without auth."""
    resp = page.request.get("/api/public/stats")
    assert resp.status == 200
    body = resp.json()
    assert "pick_ranks" in body


def test_public_homepage_loads(public_page):
    """Public homepage loads via /public prefix."""
    public_page.goto("/public")
    public_page.wait_for_load_state("domcontentloaded")
    assert public_page.title()
    assert "/login" not in public_page.url


def test_admin_requires_auth(page):
    """Admin API returns 401 without session cookie."""
    resp = page.request.get("/api/meets/")
    assert resp.status == 401


def test_admin_with_auth(auth_page):
    """Forged session cookie grants admin access."""
    auth_page.goto("/")
    # Should load the admin dashboard (not redirect to /login)
    auth_page.wait_for_load_state("domcontentloaded")
    assert "/login" not in auth_page.url


def test_csrf_token(auth_page):
    """CSRF token endpoint returns a nonce.signature token."""
    from tests.e2e.conftest import get_csrf_token
    token = get_csrf_token(auth_page)
    assert isinstance(token, str)
    assert "." in token
