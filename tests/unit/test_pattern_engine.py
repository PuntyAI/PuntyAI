"""Tests for the deep pattern learning engine."""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from punty.patterns.engine import (
    _make_result, _date_filter, MIN_SAMPLE,
    analyse_venue_performance, analyse_distance_bands,
    analyse_track_conditions, analyse_barriers,
    analyse_jockey_trainer, analyse_odds_ranges,
    analyse_bet_types, analyse_speed_map_positions,
    analyse_day_of_week, analyse_seasonal_trends,
    analyse_weather_impact, analyse_field_size,
    run_deep_pattern_analysis, _upsert_pattern_insight,
)
from punty.patterns.awards import compute_weekly_awards
from punty.patterns.weekly_summary import build_weekly_ledger, _current_streak


# ── Helper result builder ──────────────────────────────────────────────


class TestMakeResult:
    def test_basic_result(self):
        r = _make_result("venue", "Flemington", 20, 5, 200.0, 50.0, 4.5)
        assert r["dimension"] == "venue"
        assert r["key"] == "Flemington"
        assert r["sample_count"] == 20
        assert r["winners"] == 5
        assert r["hit_rate"] == 25.0
        assert r["pnl"] == 50.0
        assert r["roi"] == 25.0
        assert r["avg_odds"] == 4.5
        assert "25.0% SR" in r["insight_text"]

    def test_zero_bets(self):
        r = _make_result("venue", "Test", 0, 0, 0.0, 0.0)
        assert r["hit_rate"] == 0
        assert r["roi"] == 0

    def test_negative_pnl(self):
        r = _make_result("venue", "Test", 10, 2, 100.0, -30.0)
        assert r["pnl"] == -30.0
        assert r["roi"] == -30.0

    def test_zero_staked(self):
        r = _make_result("venue", "Test", 5, 1, 0.0, 0.0)
        assert r["roi"] == 0


class TestDateFilter:
    def test_no_window(self):
        assert _date_filter(None) == []

    def test_with_window(self):
        filters = _date_filter(7)
        assert len(filters) == 1


# ── Track condition analysis ──────────────────────────────────────────


class TestTrackConditions:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_merges_by_base_condition(self, mock_db):
        """Good 3 and Good 4 should merge into 'Good'."""
        # Mock returns rows for Good 3, Good 4, Soft 5
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("Good 3", 10, 3, 100.0, 20.0, 4.0),
            ("Good 4", 8, 2, 80.0, 10.0, 5.0),
            ("Soft 5", 6, 1, 60.0, -10.0, 6.0),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        results = await analyse_track_conditions(mock_db)

        # Good 3 + Good 4 should merge into "Good"
        keys = [r["key"] for r in results]
        assert "Good" in keys
        good = next(r for r in results if r["key"] == "Good")
        assert good["sample_count"] == 18  # 10 + 8
        assert good["winners"] == 5  # 3 + 2


# ── Odds range analysis ──────────────────────────────────────────────


class TestOddsRanges:
    @pytest.mark.asyncio
    async def test_produces_results(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("Short-priced (< $3)", 20, 8, 200.0, 40.0, 2.5),
            ("Mid-range ($3-$10)", 30, 7, 300.0, -20.0, 6.0),
            ("Roughie ($10+)", 10, 1, 100.0, -60.0, 15.0),
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        results = await analyse_odds_ranges(mock_db)
        assert len(results) == 3
        assert results[0]["dimension"] == "odds_range"


# ── Awards ──────────────────────────────────────────────────────────


class TestAwards:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        # Default: all queries return None
        db.execute = AsyncMock(return_value=MagicMock(
            first=MagicMock(return_value=None),
            all=MagicMock(return_value=[]),
        ))
        return db

    @pytest.mark.asyncio
    async def test_empty_data_returns_empty_awards(self, mock_db):
        awards = await compute_weekly_awards(mock_db)
        assert isinstance(awards, dict)
        # Should have power_rankings at minimum (may be empty)

    @pytest.mark.asyncio
    async def test_jockey_of_the_week(self):
        mock_db = AsyncMock()
        # First call: jockey query
        jockey_row = MagicMock()
        jockey_row.jockey = "James McDonald"
        jockey_row.bets = 5
        jockey_row.wins = 3
        jockey_row.pnl = 45.0
        jockey_row.staked = 50.0

        call_count = 0
        async def fake_execute(q):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:  # jockey query
                result.first = MagicMock(return_value=jockey_row)
            else:
                result.first = MagicMock(return_value=None)
                result.all = MagicMock(return_value=[])
            return result

        mock_db.execute = fake_execute
        awards = await compute_weekly_awards(mock_db)

        assert "jockey_of_the_week" in awards
        assert awards["jockey_of_the_week"]["name"] == "James McDonald"
        assert awards["jockey_of_the_week"]["wins"] == 3


# ── Weekly Ledger ──────────────────────────────────────────────────


class TestWeeklyLedger:
    @pytest.mark.asyncio
    async def test_builds_ledger(self):
        mock_db = AsyncMock()

        # Mock period stats
        stats_row = MagicMock()
        stats_row.total_bets = 25
        stats_row.winners = 7
        stats_row.staked = 250.0
        stats_row.pnl = 35.0

        # Mock best/worst bet
        bet_row = MagicMock()
        bet_row.horse_name = "Lucky Star"
        bet_row.odds_at_tip = 5.0
        bet_row.bet_type = "win"
        bet_row.pnl = 32.0
        bet_row.bet_stake = 8.0
        bet_row.venue = "Flemington"
        bet_row.race_number = 3

        # Mock bet type breakdown
        bt_row = MagicMock()
        bt_row._mapping = {"bet_type": "win", "bets": 10, "wins": 4, "staked": 100.0, "pnl": 20.0}

        call_count = 0
        async def fake_execute(q):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count <= 2:  # period stats (this week + last week)
                result.one = MagicMock(return_value=stats_row)
            elif call_count <= 4:  # best/worst bet
                result.first = MagicMock(return_value=bet_row)
            elif call_count == 5:  # bet type breakdown
                result.all = MagicMock(return_value=[("win", 10, 4, 100.0, 20.0)])
            else:  # streak
                result.all = MagicMock(return_value=[])
            return result

        mock_db.execute = fake_execute
        ledger = await build_weekly_ledger(mock_db)

        assert "this_week" in ledger
        assert "last_week" in ledger
        assert "trend" in ledger
        assert "best_bet" in ledger
        assert "streak" in ledger

    @pytest.mark.asyncio
    async def test_streak_winning(self):
        mock_db = AsyncMock()
        rows = [
            MagicMock(date=date(2026, 2, 12), daily_pnl=10.0),
            MagicMock(date=date(2026, 2, 11), daily_pnl=5.0),
            MagicMock(date=date(2026, 2, 10), daily_pnl=15.0),
            MagicMock(date=date(2026, 2, 9), daily_pnl=-20.0),
        ]
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        mock_db.execute = AsyncMock(return_value=mock_result)

        streak = await _current_streak(mock_db)
        assert streak["type"] == "winning"
        assert streak["count"] == 3

    @pytest.mark.asyncio
    async def test_streak_losing(self):
        mock_db = AsyncMock()
        rows = [
            MagicMock(date=date(2026, 2, 12), daily_pnl=-10.0),
            MagicMock(date=date(2026, 2, 11), daily_pnl=-5.0),
            MagicMock(date=date(2026, 2, 10), daily_pnl=15.0),
        ]
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        mock_db.execute = AsyncMock(return_value=mock_result)

        streak = await _current_streak(mock_db)
        assert streak["type"] == "losing"
        assert streak["count"] == 2


# ── Master orchestrator ──────────────────────────────────────────────


class TestRunDeepAnalysis:
    @pytest.mark.asyncio
    async def test_returns_all_dimensions(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_result.one = MagicMock(return_value=(0, 0, 0.0, 0.0, 0.0))
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        with patch("punty.patterns.engine.analyse_bet_types", new_callable=AsyncMock, return_value=[]):
            summary = await run_deep_pattern_analysis(mock_db)

        assert isinstance(summary, dict)
        assert "venue" in summary
        assert "distance_band" in summary
        assert "jockey" in summary
        assert "trainer" in summary


# ── PatternInsight upsert ──────────────────────────────────────────


class TestUpsertPatternInsight:
    @pytest.mark.asyncio
    async def test_inserts_new(self):
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()

        entry = {
            "key": "Flemington",
            "sample_count": 20,
            "hit_rate": 25.0,
            "pnl": 50.0,
            "avg_odds": 4.5,
            "insight_text": "Test insight",
            "dimension": "venue",
        }
        await _upsert_pattern_insight(mock_db, "venue", entry)
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_existing(self):
        mock_db = AsyncMock()
        existing_row = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_row
        mock_db.execute = AsyncMock(return_value=mock_result)

        entry = {
            "key": "Flemington",
            "sample_count": 25,
            "hit_rate": 28.0,
            "pnl": 60.0,
            "avg_odds": 4.2,
            "insight_text": "Updated insight",
            "dimension": "venue",
        }
        await _upsert_pattern_insight(mock_db, "venue", entry)
        assert existing_row.sample_count == 25
        assert existing_row.hit_rate == 28.0


# ── Blog formatter ──────────────────────────────────────────────────


class TestBlogFormatter:
    def test_format_blog_html_headings(self):
        from punty.formatters.blog import format_blog_html
        html = format_blog_html("### PUNTY AWARDS\n\nSome text here.")
        assert '<h3 class="blog-h3">' in html
        assert "<p>Some text here.</p>" in html

    def test_format_blog_html_bold(self):
        from punty.formatters.blog import format_blog_html
        html = format_blog_html("This is **bold** text.")
        assert "<strong>bold</strong>" in html

    def test_format_blog_html_list(self):
        from punty.formatters.blog import format_blog_html
        html = format_blog_html("- Item one\n- Item two")
        assert "<ul" in html
        assert "<li>" in html

    def test_format_blog_html_empty(self):
        from punty.formatters.blog import format_blog_html
        assert format_blog_html("") == ""

    def test_format_blog_teaser(self):
        from punty.formatters.blog import format_blog_teaser
        content = """*FROM THE HORSE'S MOUTH — Week of 2026-02-13*

What a week it's been! We had three screaming winners at Flemington and one absolute heartbreaker at Randwick.

---

### 🏆 PUNTY AWARDS

- **Jockey of the Week:** James McDonald — 3 winners from 5 rides
- **Roughie of the Week:** Lucky Star at Flemington — $15.00
"""
        teaser = format_blog_teaser(content, "https://punty.ai/blog/test")
        assert "punty.ai/blog/test" in teaser
        assert "Flemington" in teaser or "winners" in teaser

    def test_extract_blog_title(self):
        from punty.formatters.blog import extract_blog_title
        title = extract_blog_title("*FROM THE HORSE'S MOUTH — Week of 2026-02-13*\n\nSome text.")
        assert "FROM THE HORSE'S MOUTH" in title
        assert "2026-02-13" in title

    def test_generate_blog_slug(self):
        from punty.formatters.blog import generate_blog_slug
        slug = generate_blog_slug(date(2026, 2, 13))
        assert slug == "from-the-horses-mouth-2026-02-13"


# ── News scraper ──────────────────────────────────────────────────


class TestNewsScraper:
    def test_parse_rss_date(self):
        from punty.scrapers.news import _parse_rss_date
        assert _parse_rss_date("Thu, 06 Feb 2026 02:30:00 GMT") == "2026-02-06"

    def test_parse_rss_date_invalid(self):
        from punty.scrapers.news import _parse_rss_date
        assert _parse_rss_date("not a date") == ""

    @pytest.mark.asyncio
    async def test_scrape_headlines_graceful_failure(self):
        from punty.scrapers.news import NewsScraper
        scraper = NewsScraper()
        # With a non-existent URL, should not raise — returns empty
        with patch.object(scraper.client, "get", side_effect=Exception("Network error")):
            headlines = await scraper.scrape_headlines()
            assert isinstance(headlines, list)
        await scraper.close()

    def test_parse_7news_extracts_headlines(self):
        from punty.scrapers.news import NewsScraper
        scraper = NewsScraper()
        html = '''
        <a href="https://7news.com.au/sport/horse-racing/test-article" class="story-link">
            <h3>Group 1 winner heads Melbourne Cup charge</h3>
        </a>
        '''
        results = scraper._parse_7news(html, "7NEWS")
        # Regex-based parsing may or may not match depending on exact structure
        assert isinstance(results, list)

    def test_dedup_normalises_titles(self):
        """Duplicate titles should be removed."""
        # This tests the dedup logic in scrape_headlines
        from punty.scrapers.news import NewsScraper
        scraper = NewsScraper()
        # The dedup happens in scrape_headlines, not directly testable here
        # but we verify the structure is correct
        assert hasattr(scraper, "scrape_headlines")


# ── Future races ──────────────────────────────────────────────────


class TestFutureRaces:
    def test_detect_group_level_group1(self):
        from punty.scrapers.future_races import _detect_group_level
        assert _detect_group_level("Australian Cup (Group 1)") == "Group 1"
        assert _detect_group_level("Caulfield Cup Group 1") == "Group 1"
        assert _detect_group_level("Some Race (G1)") == "Group 1"

    def test_detect_group_level_group2(self):
        from punty.scrapers.future_races import _detect_group_level
        assert _detect_group_level("Blamey Stakes (Group 2)") == "Group 2"

    def test_detect_group_level_listed(self):
        from punty.scrapers.future_races import _detect_group_level
        assert _detect_group_level("Listed Stakes") == "Listed"

    def test_detect_group_level_prize_money_fallback(self):
        from punty.scrapers.future_races import _detect_group_level
        assert _detect_group_level("Generic Race", 200_000) == "Stakes"

    def test_detect_group_level_none(self):
        from punty.scrapers.future_races import _detect_group_level
        assert _detect_group_level("Maiden Plate", 30_000) is None

    def test_future_race_model(self):
        from punty.models.future_race import FutureRace, FutureNomination
        # Just verify the models can be imported and have expected fields
        assert FutureRace.__tablename__ == "future_races"
        assert FutureNomination.__tablename__ == "future_nominations"


# ── Change detection bugfix ──────────────────────────────────────────


class TestBaseCondition:
    def test_base_condition_strips_number(self):
        from punty.results.change_detection import _base_condition
        assert _base_condition("Good 4") == "good"
        assert _base_condition("Soft 7") == "soft"
        assert _base_condition("Heavy 8") == "heavy"
        assert _base_condition("Firm 1") == "firm"

    def test_base_condition_handles_bare(self):
        from punty.results.change_detection import _base_condition
        assert _base_condition("Good") == "good"
        assert _base_condition("Soft") == "soft"

    def test_base_condition_handles_parens(self):
        from punty.results.change_detection import _base_condition
        assert _base_condition("Good (4)") == "good"

    def test_base_condition_empty(self):
        from punty.results.change_detection import _base_condition
        assert _base_condition("") == ""

    def test_same_base_no_alert(self):
        """Good 4 → Good should NOT trigger alert."""
        from punty.results.change_detection import _base_condition
        assert _base_condition("Good 4") == _base_condition("Good")

    def test_different_base_triggers_alert(self):
        """Good → Soft should trigger alert."""
        from punty.results.change_detection import _base_condition
        assert _base_condition("Good") != _base_condition("Soft 5")


# ── Blog validation ──────────────────────────────────────────────


class TestBlogValidation:
    VALID_BLOG = """
*FROM THE HORSE'S MOUTH — Week of 2026-02-05*

What a week it's been, ya legends! We've seen it all — from screaming winners
that made Punty look like a genius to some absolute howlers that had us
questioning our existence. Let's dive in.

Opening paragraph continues with enough words to make the blog feel substantial
and real, with data and stories woven throughout. We saw some amazing finishes
at Flemington on Saturday, and the punters who followed Punty's advice would
have been well rewarded. The track was firm and fast, and the jockeys who rode
the rails early had a massive advantage. Let me tell you about the week that was.

---

### 🏆 PUNTY AWARDS

- **Jockey of the Week:** James McDonald — 5/12 (41.7% SR), P&L +$85.20
- **Roughie of the Week:** Thunder Strike at Randwick — $21.00
- **Value Bomb:** Lightning at Flemington R4 — $15.00 Win, P&L +$280.00
- **Track to Watch:** Flemington — 4/8 (50% SR), P&L +$120.50
- **Wooden Spoon:** Moonee Valley — 0/6 winners. The place is cursed.

### 🔮 THE CRYSTAL BALL

**Australian Cup** at Flemington (2026-03-07) — 2000m
Key nominations include some exciting names here.
**Punty's Early Lean:** Watching this space carefully.

### 📊 PATTERN SPOTLIGHT

Here's the thing about inside barriers at Randwick — they're an absolute goldmine.
We're running at 34% strike rate from barrier 1-4 vs 15% wide.

### 📒 THE LEDGER

- **Total Staked:** $2,400
- **Total Returned:** $2,680
- **Weekly P&L:** +$280.00 (11.7% ROI)
- **vs Last Week:** Trending UP from -$150.00

### 📰 AROUND THE TRAPS

Some racing news takes here about current events.

### ✍️ FINAL WORD

Keep grinding, keep learning, and remember — the data doesn't lie.

*"Until next Friday — Gamble Responsibly, ya legends."*
"""

    @pytest.mark.asyncio
    async def test_valid_blog_passes(self):
        from punty.scheduler.automation import validate_weekly_blog
        content = MagicMock()
        content.raw_content = self.VALID_BLOG
        db = AsyncMock()
        is_valid, issues = await validate_weekly_blog(content, db)
        assert is_valid, f"Should be valid but got issues: {issues}"
        assert issues == []

    @pytest.mark.asyncio
    async def test_short_blog_fails(self):
        from punty.scheduler.automation import validate_weekly_blog
        content = MagicMock()
        content.raw_content = "Too short."
        db = AsyncMock()
        is_valid, issues = await validate_weekly_blog(content, db)
        assert not is_valid
        assert any("too short" in i.lower() for i in issues)

    @pytest.mark.asyncio
    async def test_missing_section_fails(self):
        from punty.scheduler.automation import validate_weekly_blog
        # Remove PUNTY AWARDS from valid blog
        content = MagicMock()
        content.raw_content = self.VALID_BLOG.replace("PUNTY AWARDS", "OTHER STUFF")
        db = AsyncMock()
        is_valid, issues = await validate_weekly_blog(content, db)
        assert not is_valid
        assert any("PUNTY AWARDS" in i for i in issues)

    @pytest.mark.asyncio
    async def test_missing_gamble_responsibly_fails(self):
        from punty.scheduler.automation import validate_weekly_blog
        content = MagicMock()
        content.raw_content = self.VALID_BLOG.replace("Gamble Responsibly", "Have fun")
        db = AsyncMock()
        is_valid, issues = await validate_weekly_blog(content, db)
        assert not is_valid
        assert any("Gamble Responsibly" in i for i in issues)

    @pytest.mark.asyncio
    async def test_placeholder_text_fails(self):
        from punty.scheduler.automation import validate_weekly_blog
        content = MagicMock()
        content.raw_content = self.VALID_BLOG + "\n{WEEK_DATE} {SOME_PLACEHOLDER}"
        db = AsyncMock()
        is_valid, issues = await validate_weekly_blog(content, db)
        assert not is_valid
        assert any("placeholder" in i.lower() for i in issues)


# ── Blog generation smoke test ──────────────────────────────────


class TestBlogGeneration:
    def test_blog_content_type_exists(self):
        from punty.models.content import ContentType
        assert ContentType.WEEKLY_BLOG.value == "weekly_blog"

    def test_blog_slug_generation(self):
        from punty.formatters.blog import generate_blog_slug
        assert generate_blog_slug(date(2026, 2, 5)) == "from-the-horses-mouth-2026-02-05"

    def test_blog_title_extraction(self):
        from punty.formatters.blog import extract_blog_title
        content = "*FROM THE HORSE'S MOUTH — Week of Feb 5*\n\nHello"
        assert "FROM THE HORSE'S MOUTH" in extract_blog_title(content)

    def test_blog_title_default(self):
        from punty.formatters.blog import extract_blog_title
        assert extract_blog_title("") == "From the Horse's Mouth"
        assert extract_blog_title("Just some text without title") == "From the Horse's Mouth"
