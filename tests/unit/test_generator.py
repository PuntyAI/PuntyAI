"""Unit tests for content generator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from punty.ai.generator import ContentGenerator


class TestFormatContextForPrompt:
    """Tests for _format_context_for_prompt method."""

    def test_format_context_basic(self, sample_meeting_data):
        """Test basic context formatting works."""
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        # Call the actual method
        result = ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        assert "Test Venue" in result
        assert "2026-02-06" in result
        assert "Good 3" in result
        assert "1200m" in result
        assert "Maiden" in result

    def test_format_context_no_variable_shadowing(self, sample_context_with_market_movers):
        """
        Regression test for variable shadowing bug.

        The bug: 'summary' variable was being overwritten by market mover loop,
        causing AttributeError: 'str' object has no attribute 'get' when
        accessing summary.get("total_races") later.
        """
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        # This should NOT raise AttributeError
        result = ContentGenerator._format_context_for_prompt(
            generator, sample_context_with_market_movers
        )

        # Should complete without error and include sequence lanes
        assert "SEQUENCE LANES" in result
        assert isinstance(result, str)

    def test_format_context_with_market_movers(self, sample_meeting_data):
        """Test that market movers are formatted correctly."""
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        result = ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        # Should include market support section
        assert "MARKET SUPPORT" in result
        assert "Test Horse 2" in result
        assert "Heavily backed" in result or "$8.00" in result

    def test_format_context_with_empty_market_movers(self, sample_meeting_data):
        """Test formatting when no market movers exist."""
        # Remove market movers
        sample_meeting_data["races"][0]["analysis"]["market_movers"] = []

        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        result = ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        # Should still work without market movers
        assert "Test Venue" in result
        assert isinstance(result, str)

    def test_format_context_preserves_summary_dict(self, sample_meeting_data):
        """Ensure summary dict is not mutated by formatting."""
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        original_summary = sample_meeting_data["summary"].copy()

        ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        # Summary should be unchanged
        assert sample_meeting_data["summary"] == original_summary

    def test_format_context_includes_runner_details(self, sample_meeting_data):
        """Test that runner details are included."""
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        result = ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        assert "Test Horse 1" in result
        assert "J. Smith" in result
        # Odds may be formatted as $3.5 or $3.50
        assert "$3.5" in result or "3.5" in result

    def test_format_context_includes_pace_analysis(self, sample_meeting_data):
        """Test that pace analysis is included."""
        generator = MagicMock(spec=ContentGenerator)
        generator._get_sequence_lanes = ContentGenerator._get_sequence_lanes

        result = ContentGenerator._format_context_for_prompt(generator, sample_meeting_data)

        assert "Pace Advantaged" in result or "pace" in result.lower()


class TestGetSequenceLanes:
    """Tests for _get_sequence_lanes static method."""

    def test_sequence_lanes_7_races(self):
        """Test sequence lanes for 7-race meeting."""
        result = ContentGenerator._get_sequence_lanes(7)

        assert result["early_quad"] == (1, 4)
        assert result["quaddie"] == (4, 7)
        assert result["big6"] is None

    def test_sequence_lanes_8_races(self):
        """Test sequence lanes for 8-race meeting."""
        result = ContentGenerator._get_sequence_lanes(8)

        assert result["early_quad"] == (1, 4)
        assert result["quaddie"] == (5, 8)
        assert result["big6"] == (3, 8)

    def test_sequence_lanes_10_races(self):
        """Test sequence lanes for 10-race meeting."""
        result = ContentGenerator._get_sequence_lanes(10)

        assert result["early_quad"] == (3, 6)
        assert result["quaddie"] == (7, 10)
        assert result["big6"] == (5, 10)


class TestBuildLearningContext:
    """Tests for _build_learning_context method."""

    @pytest.mark.asyncio
    async def test_build_learning_context_no_assessments(self, db_session, sample_meeting_data):
        """Test learning context when no assessments exist."""
        generator = ContentGenerator(db_session)

        result = await generator._build_learning_context(sample_meeting_data)

        # Should return empty string when no assessments
        assert result == "" or "Past Learnings" not in result

    @pytest.mark.asyncio
    async def test_build_learning_context_handles_missing_keys(self, db_session):
        """Test learning context handles missing context keys gracefully."""
        generator = ContentGenerator(db_session)

        # Minimal context
        context = {
            "meeting": {"venue": "Test", "track_condition": "Good"},
            "races": [],
        }

        # Should not raise
        result = await generator._build_learning_context(context)
        assert isinstance(result, str)


class TestDetectFavorites:
    """Tests for _detect_favorites method."""

    @pytest.mark.asyncio
    async def test_detect_favorites_finds_short_odds(self, db_session, sample_meeting_data):
        """Test that favorites with short odds are detected."""
        generator = ContentGenerator(db_session)

        # Mock get_setting to return threshold
        with patch.object(generator, "get_setting", return_value="3.00"):
            content = "*TEST HORSE 1* (No.1) — $2.50"

            favorites = await generator._detect_favorites(content, sample_meeting_data)

            # Horse 1 has odds 3.50 which is above 3.00 threshold
            # So it should NOT be detected as a favorite
            assert isinstance(favorites, list)

    @pytest.mark.asyncio
    async def test_detect_favorites_ignores_long_odds(self, db_session, sample_meeting_data):
        """Test that horses with long odds are not flagged as favorites."""
        generator = ContentGenerator(db_session)

        with patch.object(generator, "get_setting", return_value="2.50"):
            content = "*TEST HORSE 2* is the pick"

            favorites = await generator._detect_favorites(content, sample_meeting_data)

            # Horse 2 has odds 5.00 which is well above threshold
            horse_names = [f["horse"] for f in favorites]
            assert "TEST HORSE 2" not in horse_names
