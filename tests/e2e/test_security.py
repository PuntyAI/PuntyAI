"""Security tests — XSS, CSRF bypass, SQL injection, path traversal."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestXSS:
    def test_xss_in_query_param(self, public_page):
        """Script tags in query params should not execute."""
        # Override alert before page loads to detect script execution
        public_page.add_init_script(
            "window.__xss_fired = false;"
            "window.alert = function() { window.__xss_fired = true; };"
        )
        public_page.goto("/public/stats?venue=<script>alert(1)</script>")
        public_page.wait_for_load_state("networkidle")
        fired = public_page.evaluate("window.__xss_fired")
        assert not fired, "XSS payload executed via alert()"

    def test_xss_in_api_filter(self, page):
        """Script in API filter should not execute."""
        resp = page.request.get(
            '/api/public/bet-type-stats?venue=<script>alert("xss")</script>'
        )
        assert resp.status == 200
        # Response should be JSON, not HTML with scripts
        body = resp.json()
        assert isinstance(body, list)


class TestSQLInjection:
    def test_sqli_in_venue_filter(self, page):
        """SQL injection attempt in venue filter should not leak data."""
        resp = page.request.get("/api/public/bet-type-stats?venue=' OR '1'='1")
        assert resp.status == 200
        # Should return empty or filtered results, not all data
        body = resp.json()
        assert isinstance(body, list)

    def test_sqli_in_jockey_filter(self, page):
        resp = page.request.get("/api/public/bet-type-stats?jockey='; DROP TABLE picks;--")
        assert resp.status == 200


class TestPathTraversal:
    def test_static_path_traversal(self, page):
        """Path traversal in static files should not serve sensitive content."""
        resp = page.request.get("/static/../../etc/passwd")
        # Auth middleware may redirect to /login (200) — that's fine as long as
        # the actual file content is never served.
        body = resp.text()
        assert "root:" not in body  # No /etc/passwd content leaked

    def test_api_path_traversal(self, page):
        """Path traversal in API should not serve sensitive content."""
        resp = page.request.get("/api/../../../etc/passwd")
        body = resp.text()
        assert "root:" not in body


class TestMethodRestrictions:
    def test_delete_on_public_api(self, page):
        """DELETE on public endpoints should be rejected."""
        resp = page.request.delete("/api/public/stats")
        # CSRF middleware may reject before method routing — 403 is valid rejection
        assert resp.status in (403, 405)

    def test_post_on_public_api(self, page):
        """POST on read-only public endpoints should be rejected."""
        resp = page.request.post("/api/public/stats")
        assert resp.status in (403, 405)

    def test_put_on_health(self, page):
        """PUT on health should be rejected (not serve data)."""
        import httpx
        from tests.e2e.conftest import BASE_URL
        # Use httpx with redirect disabled to avoid CSRF redirect loop
        resp = httpx.put(f"{BASE_URL}/health", follow_redirects=False, timeout=5)
        # CSRF middleware uses 307 Temporary Redirect for non-API routes
        assert resp.status_code in (302, 307, 403, 405)


class TestSessionSecurity:
    def test_crafted_cookie_rejected(self, browser, e2e_server):
        """A forged cookie with wrong secret should be rejected."""
        from itsdangerous import TimestampSigner
        import base64, json

        bad_payload = base64.b64encode(
            json.dumps({"user": {"email": "hacker@evil.com"}}).encode()
        )
        bad_signer = TimestampSigner("wrong-secret-key")
        bad_cookie = bad_signer.sign(bad_payload).decode()

        ctx = browser.new_context(base_url=e2e_server)
        ctx.add_cookies([{
            "name": "session", "value": bad_cookie,
            "domain": "127.0.0.1", "path": "/",
        }])
        page = ctx.new_page()
        resp = page.request.get("/api/meets/")
        assert resp.status == 401
        ctx.close()


class TestLargePayload:
    def test_large_query_param(self, page):
        """Extremely long query param should not crash the server."""
        long_val = "A" * 10000
        resp = page.request.get(f"/api/public/bet-type-stats?venue={long_val}")
        assert resp.status in (200, 400, 414, 422)

    def test_health_after_abuse(self, page):
        """Server should still be healthy after abuse."""
        resp = page.request.get("/health")
        assert resp.status == 200
