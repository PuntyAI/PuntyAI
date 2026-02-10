"""Public API endpoint tests — all /api/public/* and /health."""

import pytest

pytestmark = [pytest.mark.e2e]


class TestHealthEndpoint:
    def test_returns_200(self, page):
        resp = page.request.get("/health")
        assert resp.status == 200

    def test_returns_healthy_status(self, page):
        body = page.request.get("/health").json()
        assert body["status"] == "healthy"
        assert "version" in body


class TestPublicStats:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/stats")
        assert resp.status == 200

    def test_has_pick_ranks(self, page):
        body = page.request.get("/api/public/stats").json()
        assert "pick_ranks" in body
        assert isinstance(body["pick_ranks"], list)

    def test_has_winner_counts(self, page):
        body = page.request.get("/api/public/stats").json()
        assert "today_winners" in body or "alltime_winners" in body


class TestBetTypeStats:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/bet-type-stats")
        assert resp.status == 200

    def test_returns_list(self, page):
        body = page.request.get("/api/public/bet-type-stats").json()
        assert isinstance(body, list)

    def test_items_have_required_fields(self, page):
        body = page.request.get("/api/public/bet-type-stats").json()
        if body:  # May be empty if no settled picks match
            item = body[0]
            assert "category" in item
            assert "type" in item

    def test_venue_filter(self, page):
        resp = page.request.get("/api/public/bet-type-stats?venue=Flemington")
        assert resp.status == 200

    def test_today_filter(self, page):
        resp = page.request.get("/api/public/bet-type-stats?today=true")
        assert resp.status == 200


class TestFilterOptions:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/filter-options")
        assert resp.status == 200

    def test_has_expected_keys(self, page):
        body = page.request.get("/api/public/filter-options").json()
        for key in ("venues", "jockeys", "trainers", "classes", "track_conditions"):
            assert key in body
            assert isinstance(body[key], list)


class TestRecentWins:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/wins")
        assert resp.status == 200

    def test_returns_wins_array(self, page):
        body = page.request.get("/api/public/wins").json()
        assert "wins" in body
        assert isinstance(body["wins"], list)


class TestNextRace:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/next-race")
        assert resp.status == 200

    def test_has_has_next_flag(self, page):
        body = page.request.get("/api/public/next-race").json()
        assert "has_next" in body


class TestVenues:
    def test_returns_200(self, page):
        resp = page.request.get("/api/public/venues")
        assert resp.status == 200

    def test_returns_venues_array(self, page):
        body = page.request.get("/api/public/venues").json()
        assert "venues" in body
        assert isinstance(body["venues"], list)
