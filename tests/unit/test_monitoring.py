"""Tests for the monitoring module: performance tracking, alerts, regression detection."""

import secrets
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.meeting import Meeting
from punty.models.content import Content
from punty.models.pick import Pick
from punty.monitoring.performance import (
    compute_daily_digest,
    compute_rolling_comparison,
    check_regressions,
    check_calibration_drift,
    check_intraday_loss,
    DAILY_LOSS_LIMIT,
    ROI_FLOOR,
    STRIKE_RATE_DROP,
    CONSECUTIVE_LOSS_DAYS,
)
from punty.monitoring.alerts import format_daily_digest, send_daily_digest

_pick_counter = 0


def _pick_id() -> str:
    """Generate a unique pick ID for testing."""
    global _pick_counter
    _pick_counter += 1
    return f"tp{_pick_counter:012d}"


# --- Helpers ---

async def _seed_meeting_and_picks(db: AsyncSession, meeting_date: date, picks_data: list[dict]):
    """Seed a meeting with content and picks for testing."""
    meeting_id = f"test-{meeting_date.isoformat()}"

    # Check if meeting already exists (some tests seed multiple days)
    from sqlalchemy import select as sa_select
    existing = await db.execute(sa_select(Meeting).where(Meeting.id == meeting_id))
    if not existing.scalar_one_or_none():
        meeting = Meeting(
            id=meeting_id,
            venue="Test Venue",
            date=meeting_date,
        )
        db.add(meeting)

    content_id = f"content-{meeting_id}-{secrets.token_hex(4)}"
    content = Content(
        id=content_id,
        meeting_id=meeting_id,
        content_type="early_mail",
        status="approved",
        raw_content="test",
    )
    db.add(content)

    for i, pd in enumerate(picks_data):
        pick = Pick(
            id=_pick_id(),
            meeting_id=meeting_id,
            content_id=content_id,
            race_number=pd.get("race_number", i + 1),
            pick_type=pd.get("pick_type", "selection"),
            horse_name=pd.get("horse_name", f"Horse {i}"),
            saddlecloth=pd.get("saddlecloth", i + 1),
            tip_rank=pd.get("tip_rank", 1),
            bet_type=pd.get("bet_type", "win"),
            bet_stake=pd.get("bet_stake", 5.0),
            odds_at_tip=pd.get("odds_at_tip", 3.0),
            hit=pd.get("hit", False),
            pnl=pd.get("pnl", -5.0),
            settled=pd.get("settled", True),
            win_probability=pd.get("win_probability"),
            is_puntys_pick=pd.get("is_puntys_pick", False),
        )
        db.add(pick)

    await db.commit()
    return meeting_id


# --- Tests: compute_daily_digest ---

class TestComputeDailyDigest:
    async def test_empty_day(self, db_session):
        """No picks → zero totals."""
        result = await compute_daily_digest(db_session, date(2026, 2, 23))
        assert result["total_bets"] == 0
        assert result["total_pnl"] == 0

    async def test_with_settled_picks(self, db_session):
        """Settled picks are included in digest."""
        today = date(2026, 2, 23)
        await _seed_meeting_and_picks(db_session, today, [
            {"pnl": 10.0, "hit": True, "bet_stake": 5.0},
            {"pnl": -5.0, "hit": False, "bet_stake": 5.0, "race_number": 2},
            {"pnl": -5.0, "hit": False, "bet_stake": 5.0, "race_number": 3},
        ])

        result = await compute_daily_digest(db_session, today)
        assert result["total_bets"] == 3
        assert result["total_winners"] == 1
        assert result["total_pnl"] == 0.0

    async def test_unsettled_excluded(self, db_session):
        """Unsettled picks should not appear."""
        today = date(2026, 2, 23)
        await _seed_meeting_and_picks(db_session, today, [
            {"pnl": 10.0, "hit": True, "settled": True, "bet_stake": 5.0},
            {"pnl": 0.0, "hit": False, "settled": False, "bet_stake": 5.0, "race_number": 2},
        ])

        result = await compute_daily_digest(db_session, today)
        assert result["total_bets"] == 1


# --- Tests: compute_rolling_comparison ---

class TestComputeRollingComparison:
    @patch("punty.monitoring.performance.melb_today")
    async def test_rolling_comparison(self, mock_today, db_session):
        """Compare current 7 days vs previous 7 days."""
        mock_today.return_value = date(2026, 2, 23)

        # Current week: winning
        for i in range(7):
            d = date(2026, 2, 17) + timedelta(days=i)
            await _seed_meeting_and_picks(db_session, d, [
                {"pnl": 20.0, "hit": True, "bet_stake": 5.0},
            ])

        # Previous week: losing
        for i in range(7):
            d = date(2026, 2, 10) + timedelta(days=i)
            await _seed_meeting_and_picks(db_session, d, [
                {"pnl": -5.0, "hit": False, "bet_stake": 5.0},
            ])

        result = await compute_rolling_comparison(db_session)
        assert result["current_7d_pnl"] > 0
        assert result["previous_7d_pnl"] < 0
        assert result["pnl_delta"] > 0
        assert result["current_strike_rate"] > result["previous_strike_rate"]

    @patch("punty.monitoring.performance.melb_today")
    async def test_empty_weeks(self, mock_today, db_session):
        """No data → zeroes."""
        mock_today.return_value = date(2026, 2, 23)
        result = await compute_rolling_comparison(db_session)
        assert result["current_7d_pnl"] == 0
        assert result["previous_7d_pnl"] == 0


# --- Tests: check_regressions ---

class TestCheckRegressions:
    @patch("punty.monitoring.performance.melb_today")
    async def test_no_regressions(self, mock_today, db_session):
        """Good performance → no alerts."""
        mock_today.return_value = date(2026, 2, 23)
        for i in range(7):
            d = date(2026, 2, 17) + timedelta(days=i)
            await _seed_meeting_and_picks(db_session, d, [
                {"pnl": 10.0, "hit": True, "bet_stake": 5.0},
            ])

        alerts = await check_regressions(db_session)
        assert len(alerts) == 0

    @patch("punty.monitoring.performance.melb_today")
    async def test_consecutive_losses(self, mock_today, db_session):
        """3+ consecutive losing days triggers alert."""
        mock_today.return_value = date(2026, 2, 23)
        for i in range(4):
            d = date(2026, 2, 20) + timedelta(days=i)
            await _seed_meeting_and_picks(db_session, d, [
                {"pnl": -20.0, "hit": False, "bet_stake": 5.0},
            ])

        alerts = await check_regressions(db_session)
        assert any("consecutive losing days" in a for a in alerts)

    @patch("punty.monitoring.performance.melb_today")
    async def test_daily_loss_limit(self, mock_today, db_session):
        """Single day loss exceeding limit triggers alert."""
        mock_today.return_value = date(2026, 2, 23)
        # Most recent day with big loss
        await _seed_meeting_and_picks(db_session, date(2026, 2, 23), [
            {"pnl": -250.0, "hit": False, "bet_stake": 50.0, "race_number": r}
            for r in range(1, 6)
        ])

        alerts = await check_regressions(db_session)
        assert any("exceeds" in a for a in alerts)

    @patch("punty.monitoring.performance.melb_today")
    async def test_empty_history(self, mock_today, db_session):
        """No data → no alerts."""
        mock_today.return_value = date(2026, 2, 23)
        alerts = await check_regressions(db_session)
        assert len(alerts) == 0


# --- Tests: check_calibration_drift ---

class TestCheckCalibrationDrift:
    @patch("punty.monitoring.performance.melb_today")
    async def test_insufficient_data(self, mock_today, db_session):
        """Less than 50 picks → returns None."""
        mock_today.return_value = date(2026, 2, 23)
        result = await check_calibration_drift(db_session)
        assert result is None

    @patch("punty.monitoring.performance.melb_today")
    async def test_well_calibrated(self, mock_today, db_session):
        """Predictions matching outcomes → no drift."""
        mock_today.return_value = date(2026, 2, 23)
        # Seed 60 picks at ~30% probability, ~30% hit rate
        picks = []
        for i in range(60):
            picks.append({
                "win_probability": 0.30,
                "hit": i < 18,  # 30% hit rate
                "pnl": 10.0 if i < 18 else -5.0,
                "bet_stake": 5.0,
                "race_number": (i % 8) + 1,
            })
        await _seed_meeting_and_picks(db_session, date(2026, 2, 22), picks[:8])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 21), picks[8:16])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 20), picks[16:24])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 19), picks[24:32])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 18), picks[32:40])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 17), picks[40:48])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 16), picks[48:56])
        await _seed_meeting_and_picks(db_session, date(2026, 2, 15), picks[56:])

        result = await check_calibration_drift(db_session)
        # With well-calibrated data, should not flag drift
        assert result is None

    @patch("punty.monitoring.performance.melb_today")
    async def test_detects_overconfidence(self, mock_today, db_session):
        """High predicted prob but low hit rate → overconfident drift."""
        mock_today.return_value = date(2026, 2, 23)
        # 60 picks predicted at 50% but only 20% actually hit
        picks = []
        for i in range(60):
            picks.append({
                "win_probability": 0.50,
                "hit": i < 12,  # 20% hit rate
                "pnl": 10.0 if i < 12 else -5.0,
                "bet_stake": 5.0,
                "race_number": (i % 8) + 1,
            })
        # Spread across multiple days
        for j in range(8):
            d = date(2026, 2, 15) + timedelta(days=j)
            start = j * 8
            end = min(start + 8, 60)
            if start < 60:
                await _seed_meeting_and_picks(db_session, d, picks[start:end])

        result = await check_calibration_drift(db_session)
        assert result is not None
        assert "overconfident" in result


# --- Tests: check_intraday_loss ---

class TestCheckIntradayLoss:
    @patch("punty.monitoring.performance.melb_today")
    async def test_no_breach(self, mock_today, db_session):
        """Normal day → no alert."""
        mock_today.return_value = date(2026, 2, 23)
        await _seed_meeting_and_picks(db_session, date(2026, 2, 23), [
            {"pnl": -10.0, "hit": False, "bet_stake": 5.0},
        ])
        result = await check_intraday_loss(db_session)
        assert result is None

    @patch("punty.monitoring.performance.melb_today")
    async def test_breach(self, mock_today, db_session):
        """Big loss day → alert."""
        mock_today.return_value = date(2026, 2, 23)
        await _seed_meeting_and_picks(db_session, date(2026, 2, 23), [
            {"pnl": -80.0, "hit": False, "bet_stake": 20.0, "race_number": r}
            for r in range(1, 5)
        ])
        result = await check_intraday_loss(db_session)
        assert result is not None
        assert "loss alert" in result.lower() or "Daily loss" in result


# --- Tests: format_daily_digest ---

class TestFormatDailyDigest:
    def test_basic_format(self):
        """Formats a complete digest message."""
        digest = {
            "date": "2026-02-23",
            "total_bets": 12,
            "total_winners": 5,
            "total_strike_rate": 41.7,
            "total_staked": 240.0,
            "total_returned": 287.5,
            "total_pnl": 47.50,
            "by_product": {
                "selection": {"bets": 8, "winners": 3, "strike_rate": 37.5, "staked": 160.0, "pnl": 32.0},
                "exotic": {"bets": 2, "winners": 1, "strike_rate": 50.0, "staked": 40.0, "pnl": 15.5},
                "sequence": {"bets": 2, "winners": 1, "strike_rate": 50.0, "staked": 40.0, "pnl": 0.0},
            },
        }
        comparison = {
            "current_7d_pnl": 156.0,
            "previous_7d_pnl": 89.0,
            "pnl_delta": 67.0,
            "current_7d_bets": 84,
            "previous_7d_bets": 72,
            "current_strike_rate": 38.2,
            "previous_strike_rate": 35.1,
            "sr_delta": 3.1,
        }
        regressions = []
        calibration = None

        msg = format_daily_digest(digest, comparison, regressions, calibration)
        assert "Daily P&L Report" in msg
        assert "+$47.50" in msg
        assert "12 bets" in msg
        assert "Selections" in msg
        assert "Rolling 7-Day" in msg
        assert "No regressions detected" in msg

    def test_with_regressions(self):
        """Regression warnings appear in message."""
        digest = {
            "date": "2026-02-23",
            "total_bets": 5,
            "total_winners": 0,
            "total_strike_rate": 0.0,
            "total_staked": 100.0,
            "total_returned": 0.0,
            "total_pnl": -100.0,
            "by_product": {},
        }
        comparison = {
            "current_7d_pnl": -300.0,
            "previous_7d_pnl": 50.0,
            "pnl_delta": -350.0,
            "current_7d_bets": 50,
            "previous_7d_bets": 50,
            "current_strike_rate": 10.0,
            "previous_strike_rate": 35.0,
            "sr_delta": -25.0,
        }
        regressions = ["7-day ROI at -30.0% (below -15% floor)", "3 consecutive losing days"]
        calibration = None

        msg = format_daily_digest(digest, comparison, regressions, calibration)
        assert "REGRESSIONS" in msg
        assert "7-day ROI" in msg
        assert "consecutive" in msg

    def test_negative_pnl_formatting(self):
        """Negative P&L should not have + sign."""
        digest = {
            "date": "2026-02-23",
            "total_bets": 1,
            "total_winners": 0,
            "total_strike_rate": 0.0,
            "total_staked": 20.0,
            "total_returned": 0.0,
            "total_pnl": -20.0,
            "by_product": {},
        }
        comparison = {
            "current_7d_pnl": 0, "previous_7d_pnl": 0, "pnl_delta": 0,
            "current_7d_bets": 0, "previous_7d_bets": 0,
            "current_strike_rate": 0, "previous_strike_rate": 0, "sr_delta": 0,
        }
        msg = format_daily_digest(digest, comparison, [], None)
        assert "-$20.00" in msg
        assert "+$-" not in msg
        assert "$-" not in msg


# --- Tests: send_daily_digest ---

class TestSendDailyDigest:
    @patch("punty.monitoring.alerts.async_session")
    @patch("punty.monitoring.alerts.melb_today")
    async def test_skips_when_no_bot(self, mock_today, mock_session):
        """No Telegram bot → skip silently."""
        app = MagicMock()
        app.state.telegram_bot = None
        await send_daily_digest(app)
        mock_session.assert_not_called()

    @patch("punty.monitoring.alerts.async_session")
    @patch("punty.monitoring.alerts.melb_today")
    async def test_skips_when_no_bets(self, mock_today, mock_session):
        """No bets today → skip."""
        mock_today.return_value = date(2026, 2, 23)

        mock_db = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.return_value = mock_ctx

        app = MagicMock()
        bot = MagicMock()
        bot.is_running.return_value = True
        bot.send_alert = AsyncMock()
        app.state.telegram_bot = bot

        with patch("punty.monitoring.alerts.compute_daily_digest", new_callable=AsyncMock) as mock_digest:
            mock_digest.return_value = {"total_bets": 0, "total_pnl": 0}
            await send_daily_digest(app)
            bot.send_alert.assert_not_called()
