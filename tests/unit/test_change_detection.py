"""Tests for pre-race change detection (scratchings, track, jockey/gear)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from punty.results.change_detection import (
    ChangeAlert,
    MeetingSnapshot,
    RunnerSnapshot,
    detect_scratching_changes,
    detect_track_condition_change,
    detect_jockey_gear_changes,
    find_impacted_picks,
    find_alternative,
    compose_scratching_alert,
    compose_track_alert,
    compose_jockey_alert,
    compose_gear_alert,
    take_snapshot,
    _normalise_condition,
    _normalise_name,
    _horse_matches,
    _extract_track_number,
)


# ── Helper factories ────────────────────────────────────────────────────────


def _make_runner(
    race_id="meet-r5",
    horse_name="Star Runner",
    saddlecloth=3,
    scratched=False,
    jockey="J. Smith",
    gear="Blinkers",
    gear_changes=None,
    current_odds=4.50,
    speed_map_position="on_pace",
):
    runner = MagicMock()
    runner.race_id = race_id
    runner.horse_name = horse_name
    runner.saddlecloth = saddlecloth
    runner.scratched = scratched
    runner.jockey = jockey
    runner.gear = gear
    runner.gear_changes = gear_changes
    runner.current_odds = current_odds
    runner.speed_map_position = speed_map_position
    return runner


def _make_pick(
    pick_id="p1",
    meeting_id="meet",
    race_number=5,
    horse_name="Star Runner",
    saddlecloth=3,
    tip_rank=1,
    pick_type="selection",
    bet_type="win",
    bet_stake=8.0,
    settled=False,
    exotic_type=None,
    exotic_runners=None,
    sequence_type=None,
    sequence_legs=None,
    sequence_start_race=None,
    sequence_variant=None,
):
    pick = MagicMock()
    pick.id = pick_id
    pick.meeting_id = meeting_id
    pick.race_number = race_number
    pick.horse_name = horse_name
    pick.saddlecloth = saddlecloth
    pick.tip_rank = tip_rank
    pick.pick_type = pick_type
    pick.bet_type = bet_type
    pick.bet_stake = bet_stake
    pick.settled = settled
    pick.exotic_type = exotic_type
    pick.exotic_runners = exotic_runners
    pick.sequence_type = sequence_type
    pick.sequence_legs = sequence_legs
    pick.sequence_start_race = sequence_start_race
    pick.sequence_variant = sequence_variant
    return pick


def _snapshot(runners_dict=None, track_condition="Good 4"):
    """Build a MeetingSnapshot from a dict of {key: RunnerSnapshot}."""
    snap = MeetingSnapshot(track_condition=track_condition)
    if runners_dict:
        snap.runners = runners_dict
    return snap


# ── Unit tests: helpers ─────────────────────────────────────────────────────


class TestHelpers:
    def test_normalise_condition(self):
        assert _normalise_condition("Good 4") == "good 4"
        assert _normalise_condition("Soft(5)") == "soft5"
        assert _normalise_condition("  Heavy 10 ") == "heavy 10"

    def test_normalise_name(self):
        assert _normalise_name("J. Smith") == "j smith"
        assert _normalise_name("J Smith") == "j smith"
        assert _normalise_name("  Craig Newitt ") == "craig newitt"

    def test_horse_matches_by_saddlecloth(self):
        pick = _make_pick(saddlecloth=5)
        assert _horse_matches(pick, "Different Horse", 5) is True

    def test_horse_matches_by_name(self):
        pick = _make_pick(horse_name="Star Runner", saddlecloth=None)
        assert _horse_matches(pick, "Star Runner", 99) is True

    def test_horse_matches_case_insensitive(self):
        pick = _make_pick(horse_name="STAR RUNNER", saddlecloth=None)
        assert _horse_matches(pick, "star runner", 99) is True

    def test_horse_no_match(self):
        pick = _make_pick(horse_name="Other Horse", saddlecloth=7)
        assert _horse_matches(pick, "Star Runner", 3) is False

    def test_extract_track_number(self):
        assert _extract_track_number("Good 4") == 4
        assert _extract_track_number("Soft 7") == 7
        assert _extract_track_number("Heavy") is None

    def test_dedup_key_scratching(self):
        alert = ChangeAlert(
            change_type="scratching", meeting_id="m", race_number=5, horse_name="Star"
        )
        assert alert.dedup_key == "scratching:R5:Star"

    def test_dedup_key_track(self):
        alert = ChangeAlert(
            change_type="track_condition", meeting_id="m", old_value="Good 4", new_value="Soft 5"
        )
        assert alert.dedup_key == "track:Good 4->Soft 5"


# ── Unit tests: message composition ─────────────────────────────────────────


class TestMessageComposition:
    def test_scratching_alert_selection(self):
        picks = [{"pick_type": "selection", "tip_rank": 2, "bet_type": "win", "bet_stake": 6}]
        alt = {"horse_name": "Plan B", "odds": 5.50, "speed_map_position": "on_pace"}
        msg = compose_scratching_alert("Star Runner", 5, picks, alt)
        assert "SCRATCHING" in msg
        assert "Star Runner" in msg
        assert "#2 pick" in msg
        assert "R5" in msg
        assert "Plan B" in msg
        assert "$5.50" in msg
        assert len(msg) <= 280

    def test_scratching_alert_exotic(self):
        picks = [{"pick_type": "exotic", "exotic_type": "Trifecta", "remaining_runners": 3, "total_runners": 4}]
        msg = compose_scratching_alert("Gone Horse", 3, picks)
        assert "SCRATCHING" in msg
        assert "Trifecta" in msg
        assert "3 of 4" in msg

    def test_scratching_alert_sequence(self):
        picks = [{"pick_type": "sequence", "sequence_type": "quaddie", "sequence_variant": "skinny",
                  "leg_number": 2, "remaining_in_leg": 1, "total_in_leg": 2}]
        msg = compose_scratching_alert("Quad Horse", 7, picks)
        assert "SCRATCHING" in msg
        assert "Leg 2" in msg
        assert "1 runner" in msg

    def test_scratching_alert_no_picks(self):
        msg = compose_scratching_alert("Random Horse", 4, [])
        assert "SCRATCHING" in msg
        assert "Random Horse" in msg
        assert "R4" in msg

    def test_track_alert_upgrade(self):
        msg = compose_track_alert("Flemington", "Soft 5", "Good 3")
        assert "TRACK UPDATE" in msg
        assert "Flemington" in msg
        assert "Soft 5" in msg
        assert "Good 3" in msg
        assert len(msg) <= 280

    def test_track_alert_downgrade(self):
        msg = compose_track_alert("Doomben", "Good 4", "Soft 7")
        assert "TRACK UPDATE" in msg
        assert "Doomben" in msg

    def test_jockey_alert(self):
        msg = compose_jockey_alert("Star Runner", 5, "J. McDonald", "C. Williams", tip_rank=1)
        assert "JOCKEY CHANGE" in msg
        assert "Star Runner" in msg
        assert "#1 pick" in msg
        assert "J. McDonald" in msg
        assert "C. Williams" in msg
        assert len(msg) <= 280

    def test_jockey_alert_no_rank(self):
        msg = compose_jockey_alert("Horse", 3, "Old Jock", "New Jock")
        assert "#" not in msg.split("JOCKEY")[1]  # No tip rank

    def test_gear_alert(self):
        msg = compose_gear_alert("Speedy", 6, "Blinkers ON first time", tip_rank=3)
        assert "GEAR CHANGE" in msg
        assert "Speedy" in msg
        assert "#3 pick" in msg
        assert "Blinkers ON" in msg
        assert len(msg) <= 280

    def test_messages_truncate_at_280(self):
        # Very long horse name + alternative
        picks = [{"pick_type": "selection", "tip_rank": 1, "bet_type": "win", "bet_stake": 10}]
        alt = {"horse_name": "A" * 200, "odds": 99.99, "speed_map_position": "backmarker"}
        msg = compose_scratching_alert("B" * 200, 5, picks, alt)
        assert len(msg) <= 280


# ── Unit tests: pick impact analysis ────────────────────────────────────────


class TestFindImpactedPicks:
    @pytest.mark.asyncio
    async def test_finds_selection_by_saddlecloth(self):
        pick = _make_pick(saddlecloth=3, pick_type="selection", tip_rank=1)
        db = AsyncMock()

        # Three queries: selections, exotics, sequences
        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [pick]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        picks = await find_impacted_picks(db, "meet", 5, "Star Runner", 3)
        assert len(picks) == 1
        assert picks[0]["pick_type"] == "selection"
        assert picks[0]["tip_rank"] == 1

    @pytest.mark.asyncio
    async def test_finds_exotic_by_saddlecloth_in_json(self):
        pick = _make_pick(
            pick_type="exotic", exotic_type="Trifecta",
            exotic_runners=json.dumps([1, 3, 5, 7])
        )
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 2:  # exotics query
                result.scalars.return_value.all.return_value = [pick]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        picks = await find_impacted_picks(db, "meet", 5, "Star Runner", 3)
        assert len(picks) == 1
        assert picks[0]["pick_type"] == "exotic"
        assert picks[0]["remaining_runners"] == 3
        assert picks[0]["total_runners"] == 4

    @pytest.mark.asyncio
    async def test_finds_sequence_leg(self):
        pick = _make_pick(
            pick_type="sequence", sequence_type="quaddie",
            sequence_legs=json.dumps([[1, 3], [2, 5], [4], [1, 6]]),
            sequence_start_race=5, sequence_variant="skinny",
        )
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 3:  # sequences query
                result.scalars.return_value.all.return_value = [pick]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        # Saddlecloth 3 is in leg 1 (race 5 - start_race 5 = index 0)
        picks = await find_impacted_picks(db, "meet", 5, "Star Runner", 3)
        assert len(picks) == 1
        assert picks[0]["pick_type"] == "sequence"
        assert picks[0]["leg_number"] == 1
        assert picks[0]["remaining_in_leg"] == 1  # leg was [1, 3], 3 removed = [1]

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        picks = await find_impacted_picks(db, "meet", 5, "Nobody", 99)
        assert picks == []


# ── Unit tests: alternative finder ──────────────────────────────────────────


class TestFindAlternative:
    @pytest.mark.asyncio
    async def test_finds_best_alternative(self):
        runner = _make_runner(horse_name="Plan B", saddlecloth=7, current_odds=3.50)
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Existing pick saddlecloths
                result.all.return_value = [(3,)]
            else:
                result.scalars.return_value.all.return_value = [runner]
            return result
        db.execute = mock_execute

        alt = await find_alternative(db, "meet", 5, 3)
        assert alt is not None
        assert alt["horse_name"] == "Plan B"
        assert alt["odds"] == 3.50

    @pytest.mark.asyncio
    async def test_excludes_existing_picks(self):
        runner_picked = _make_runner(horse_name="Already Picked", saddlecloth=5, current_odds=2.0)
        runner_alt = _make_runner(horse_name="Fresh Pick", saddlecloth=9, current_odds=6.0)
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.all.return_value = [(5,)]  # saddlecloth 5 already picked
            else:
                result.scalars.return_value.all.return_value = [runner_picked, runner_alt]
            return result
        db.execute = mock_execute

        alt = await find_alternative(db, "meet", 5, 3)
        assert alt is not None
        assert alt["horse_name"] == "Fresh Pick"

    @pytest.mark.asyncio
    async def test_no_alternatives_returns_none(self):
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.all.return_value = []
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        alt = await find_alternative(db, "meet", 5, 3)
        assert alt is None


# ── Unit tests: detection functions ─────────────────────────────────────────


class TestDetectScratchingChanges:
    @pytest.mark.asyncio
    async def test_detects_new_scratching(self):
        scratched_runner = _make_runner(scratched=True, saddlecloth=3, horse_name="Star Runner")
        db = AsyncMock()

        # Mocking: first call for scratched runners, then pick impact queries
        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [scratched_runner]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        snapshot = _snapshot({
            "meet-r5:3": RunnerSnapshot(5, "Star Runner", 3, False, "J. Smith", None, None)
        })

        alerts = await detect_scratching_changes(db, "meet", [5], snapshot)
        assert len(alerts) == 1
        assert alerts[0].change_type == "scratching"
        assert alerts[0].horse_name == "Star Runner"

    @pytest.mark.asyncio
    async def test_ignores_already_scratched(self):
        scratched_runner = _make_runner(scratched=True, saddlecloth=3)
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [scratched_runner]
        db.execute = AsyncMock(return_value=result_mock)

        # Runner was already scratched in snapshot
        snapshot = _snapshot({
            "meet-r5:3": RunnerSnapshot(5, "Star Runner", 3, True, "J. Smith", None, None)
        })

        alerts = await detect_scratching_changes(db, "meet", [5], snapshot)
        assert len(alerts) == 0


class TestDetectTrackConditionChange:
    @pytest.mark.asyncio
    async def test_detects_upgrade(self):
        meeting = MagicMock()
        meeting.track_condition = "Good 3"
        meeting.venue = "Flemington"

        db = AsyncMock()
        db.get = AsyncMock(return_value=meeting)

        snapshot = _snapshot(track_condition="Soft 5")
        alert = await detect_track_condition_change(db, "meet", snapshot)
        assert alert is not None
        assert alert.change_type == "track_condition"
        assert "TRACK UPDATE" in alert.message

    @pytest.mark.asyncio
    async def test_no_change(self):
        meeting = MagicMock()
        meeting.track_condition = "Good 4"

        db = AsyncMock()
        db.get = AsyncMock(return_value=meeting)

        snapshot = _snapshot(track_condition="Good 4")
        alert = await detect_track_condition_change(db, "meet", snapshot)
        assert alert is None

    @pytest.mark.asyncio
    async def test_no_change_format_difference(self):
        """Format differences like 'Good (4)' vs 'Good 4' should NOT trigger alerts."""
        meeting = MagicMock()
        meeting.track_condition = "Good (4)"

        db = AsyncMock()
        db.get = AsyncMock(return_value=meeting)

        snapshot = _snapshot(track_condition="Good 4")
        alert = await detect_track_condition_change(db, "meet", snapshot)
        assert alert is None

    @pytest.mark.asyncio
    async def test_no_change_case_difference(self):
        """Case differences like 'GOOD 4' vs 'Good 4' should NOT trigger alerts."""
        meeting = MagicMock()
        meeting.track_condition = "GOOD 4"

        db = AsyncMock()
        db.get = AsyncMock(return_value=meeting)

        snapshot = _snapshot(track_condition="Good 4")
        alert = await detect_track_condition_change(db, "meet", snapshot)
        assert alert is None


class TestDetectJockeyGearChanges:
    @pytest.mark.asyncio
    async def test_detects_jockey_change_on_pick(self):
        runner = _make_runner(saddlecloth=3, jockey="C. Williams", scratched=False)
        pick = _make_pick(saddlecloth=3, tip_rank=1)
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Runners query
                result.scalars.return_value.all.return_value = [runner]
            elif call_count == 2:
                # Impacted picks — selections
                result.scalars.return_value.all.return_value = [pick]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        snapshot = _snapshot({
            "meet-r5:3": RunnerSnapshot(5, "Star Runner", 3, False, "J. McDonald", None, None)
        })

        alerts = await detect_jockey_gear_changes(db, "meet", [5], snapshot)
        jockey_alerts = [a for a in alerts if a.change_type == "jockey_change"]
        assert len(jockey_alerts) == 1
        assert "J. McDonald" in jockey_alerts[0].old_value
        assert "C. Williams" in jockey_alerts[0].new_value

    @pytest.mark.asyncio
    async def test_ignores_jockey_change_on_non_pick(self):
        runner = _make_runner(saddlecloth=3, jockey="New Jock", scratched=False)
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [runner]
            else:
                result.scalars.return_value.all.return_value = []  # No picks
            return result
        db.execute = mock_execute

        snapshot = _snapshot({
            "meet-r5:3": RunnerSnapshot(5, "Star Runner", 3, False, "Old Jock", None, None)
        })

        alerts = await detect_jockey_gear_changes(db, "meet", [5], snapshot)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_detects_gear_change_on_pick(self):
        runner = _make_runner(saddlecloth=3, gear_changes="Blinkers ON", scratched=False)
        pick = _make_pick(saddlecloth=3, tip_rank=2)
        db = AsyncMock()

        call_count = 0
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.scalars.return_value.all.return_value = [runner]
            elif call_count == 2:
                # selection picks for gear check
                result.scalars.return_value.all.return_value = [pick]
            else:
                result.scalars.return_value.all.return_value = []
            return result
        db.execute = mock_execute

        snapshot = _snapshot({
            "meet-r5:3": RunnerSnapshot(5, "Star Runner", 3, False, "J. Smith", "Blinkers", None)
        })

        alerts = await detect_jockey_gear_changes(db, "meet", [5], snapshot)
        gear_alerts = [a for a in alerts if a.change_type == "gear_change"]
        assert len(gear_alerts) == 1
        assert "Blinkers ON" in gear_alerts[0].message
