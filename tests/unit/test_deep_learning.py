"""Tests for the deep learning pattern engine."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from punty.deep_learning.models import (
    Base,
    HistoricalRace,
    HistoricalRunner,
    HistoricalSectional,
    init_db,
    get_session,
)
from punty.deep_learning.importer import (
    _parse_filename,
    _parse_in_run,
    _parse_flucs,
    _parse_time_to_secs,
    _resolve_state,
    _record_to_json,
    _build_form_history,
    import_form_file,
    import_sectionals_file,
)
from punty.deep_learning.patterns import (
    _binomial_p_value,
    _distance_bucket,
    _barrier_bucket,
    _condition_group,
    _confidence_label,
    Pattern,
    analyse_track_distance_condition,
    analyse_barrier_bias,
    analyse_pace_patterns,
    analyse_form_cycles,
    analyse_class_transitions,
    analyse_condition_specialists,
    analyse_coming_into_form,
    analyse_class_movers,
    analyse_market_efficiency,
    run_all_analyses,
)


# ──── Fixtures ────


@pytest.fixture
def db_path(tmp_path):
    """Temporary SQLite DB for tests."""
    path = tmp_path / "test_deep_learning.db"
    return path


@pytest.fixture
def session(db_path):
    """Get a session with empty tables."""
    engine = init_db(db_path)
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()


def _make_race(session, race_id=1000, venue="Flemington", meeting_date=None,
               distance=1600, condition="G4", state="VIC", field_size=10,
               prize_money=100000, location_type="M"):
    """Helper: create a race and return it."""
    race = HistoricalRace(
        race_id=race_id,
        meeting_date=meeting_date or date(2025, 6, 15),
        venue=venue,
        state=state,
        country="AUS",
        location_type=location_type,
        race_number=1,
        distance=distance,
        race_class="Open",
        track_condition=condition,
        field_size=field_size,
        prize_money=prize_money,
    )
    session.add(race)
    session.flush()
    return race


def _make_runner(session, race, tab_no=1, position=1, margin=0.0,
                 barrier=1, sp=3.0, settle_pos=None, prep_runs=None,
                 jockey="J Smith", trainer="T Jones",
                 form_history=None, good_record=None,
                 opening_odds=None, last_10=None):
    """Helper: create a runner and return it."""
    runner = HistoricalRunner(
        race_fk=race.id,
        race_id=race.race_id,
        runner_id=tab_no * 100 + race.race_id,
        horse_name=f"Horse_{tab_no}",
        tab_no=tab_no,
        barrier=barrier,
        weight=58.0,
        age=4,
        sex="Gelding",
        jockey=jockey,
        trainer=trainer,
        career_starts=20,
        career_wins=4,
        win_pct=20.0,
        starting_price=sp,
        opening_odds=opening_odds,
        settle_pos=settle_pos,
        prep_runs=prep_runs,
        finish_position=position,
        margin=margin,
        won=position == 1,
        placed=position is not None and 1 <= position <= 3,
        form_history=json.dumps(form_history) if form_history else None,
        good_record=json.dumps(good_record) if good_record else None,
        last_10=last_10,
    )
    session.add(runner)
    session.flush()
    return runner


# ──── Importer helpers ────


class TestParseFilename:
    def test_standard_format(self):
        d, v = _parse_filename("250101_Flemington.json")
        assert d == date(2025, 1, 1)
        assert v == "Flemington"

    def test_multi_word_venue(self):
        d, v = _parse_filename("250615_Eagle_Farm.json")
        assert d == date(2025, 6, 15)
        assert v == "Eagle Farm"

    def test_invalid_date(self):
        d, v = _parse_filename("bad_file.json")
        assert d is None

    def test_february_date(self):
        d, v = _parse_filename("260213_Kilmore.json")
        assert d == date(2026, 2, 13)
        assert v == "Kilmore"


class TestParseInRun:
    def test_full_parse(self):
        result = _parse_in_run("settling_down,2;m800,3;m400,1;finish,1;")
        assert result == {"settle": 2, "m800": 3, "m400": 1}

    def test_empty(self):
        assert _parse_in_run(None) == {"settle": None, "m800": None, "m400": None}

    def test_partial(self):
        result = _parse_in_run("settling_down,5;")
        assert result["settle"] == 5
        assert result["m800"] is None


class TestParseFlucs:
    def test_full_parse(self):
        result = _parse_flucs("opening,2.40;mid,2.30;starting,2.60;")
        assert result["opening"] == 2.4
        assert result["mid"] == 2.3
        assert result["starting"] == 2.6

    def test_empty(self):
        result = _parse_flucs(None)
        assert result["opening"] is None


class TestParseTimeToSecs:
    def test_standard(self):
        secs = _parse_time_to_secs("00:01:34.2000000")
        assert abs(secs - 94.2) < 0.01

    def test_none(self):
        assert _parse_time_to_secs(None) is None


class TestResolveState:
    def test_from_track_data(self):
        assert _resolve_state("Flemington", {"State": "VIC"}) == "VIC"

    def test_from_venue_map(self):
        assert _resolve_state("Flemington") == "VIC"
        assert _resolve_state("Randwick") == "NSW"
        assert _resolve_state("Eagle Farm") == "QLD"

    def test_sponsor_prefix_stripped(self):
        assert _resolve_state("Sportsbet Pakenham") == "VIC"
        assert _resolve_state("Ladbrokes Geelong") == "VIC"

    def test_park_stripped(self):
        assert _resolve_state("park Kilmore") == "VIC"


class TestRecordToJson:
    def test_with_data(self):
        record = {"Starts": 5, "Firsts": 2, "Seconds": 1, "Thirds": 0}
        result = _record_to_json(record)
        assert result is not None
        assert json.loads(result)["Starts"] == 5

    def test_empty_record(self):
        assert _record_to_json({"Starts": 0}) is None
        assert _record_to_json(None) is None


class TestBuildFormHistory:
    def test_builds_compact_history(self):
        forms = [{
            "MeetingDate": "2025-06-01T00:00:00",
            "Track": {"Name": "Flemington", "State": "VIC"},
            "Distance": 1600,
            "RaceClass": "Open",
            "TrackCondition": "G4",
            "Position": 2,
            "Margin": 1.5,
            "PriceSP": 4.0,
            "Starters": 10,
            "InRun": "settling_down,3;m800,3;m400,2;finish,2;",
            "OfficialRaceTime": "00:01:36.0000000",
            "PrepRuns": 1,
            "Weight": 57.0,
            "Barrier": 3,
        }]
        result = _build_form_history(forms)
        assert result is not None
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["venue"] == "Flemington"
        assert data[0]["pos"] == 2

    def test_empty(self):
        assert _build_form_history([]) is None


# ──── Pattern helpers ────


class TestDistanceBucket:
    def test_sprint(self):
        assert _distance_bucket(1000) == "sprint"
        assert _distance_bucket(1100) == "sprint"

    def test_short(self):
        assert _distance_bucket(1200) == "short"
        assert _distance_bucket(1300) == "short"

    def test_middle(self):
        assert _distance_bucket(1400) == "middle"
        assert _distance_bucket(1600) == "middle"

    def test_staying(self):
        assert _distance_bucket(2000) == "staying"

    def test_extreme(self):
        assert _distance_bucket(2400) == "extreme"

    def test_none(self):
        assert _distance_bucket(None) == "unknown"


class TestBarrierBucket:
    def test_inside(self):
        assert _barrier_bucket(1, 12) == "inside"
        assert _barrier_bucket(3, 12) == "inside"

    def test_middle(self):
        assert _barrier_bucket(5, 12) == "middle"

    def test_outside(self):
        assert _barrier_bucket(10, 12) == "outside"

    def test_none(self):
        assert _barrier_bucket(None, 12) == "unknown"


class TestConditionGroup:
    def test_good(self):
        assert _condition_group("G4") == "Good"
        assert _condition_group("Good 3") == "Good"

    def test_soft(self):
        assert _condition_group("S5") == "Soft"

    def test_heavy(self):
        assert _condition_group("H8") == "Heavy"

    def test_firm(self):
        assert _condition_group("F1") == "Firm"

    def test_synthetic(self):
        assert _condition_group("Synthetic") == "Synthetic"


class TestBinomialPValue:
    def test_expected_rate(self):
        # 50 wins out of 100 with expected 50% should be non-significant
        p = _binomial_p_value(50, 100, 0.5)
        assert p > 0.05

    def test_significant(self):
        # 70 wins out of 100 with expected 50% should be significant
        p = _binomial_p_value(70, 100, 0.5)
        assert p < 0.01

    def test_edge_cases(self):
        assert _binomial_p_value(0, 0, 0.5) == 1.0
        assert _binomial_p_value(5, 10, 0) == 1.0


class TestConfidenceLabel:
    def test_high(self):
        assert _confidence_label(0.005, 60) == "HIGH"

    def test_medium(self):
        assert _confidence_label(0.03, 35) == "MEDIUM"

    def test_low(self):
        assert _confidence_label(0.10, 100) == "LOW"


# ──── Pattern analyses with real data ────


class TestFormCycles:
    def test_prep_run_patterns(self, session):
        """First-up runners should show different win rate to deep-prep runners."""
        race = _make_race(session, race_id=1)
        # Create 25 first-up winners and 25 first-up losers
        for i in range(25):
            _make_runner(session, race, tab_no=i+1, position=1, prep_runs=0)
        for i in range(25):
            _make_runner(session, race, tab_no=i+26, position=5, prep_runs=0)
        # Create 10 deep-prep winners and 40 losers
        race2 = _make_race(session, race_id=2)
        for i in range(10):
            _make_runner(session, race2, tab_no=i+1, position=1, prep_runs=5)
        for i in range(40):
            _make_runner(session, race2, tab_no=i+11, position=6, prep_runs=5)
        session.commit()

        patterns = analyse_form_cycles(session)
        # Should find at least prep_0 and prep_5
        assert len(patterns) >= 1
        # First-up (50% SR) should be significant vs base
        prep0 = [p for p in patterns if "prep_0" in p.dimension]
        assert len(prep0) >= 1


class TestComingIntoForm:
    def test_improving_trend_detected(self, session):
        """Horses with improving last 3 runs should be flagged."""
        race = _make_race(session, race_id=1, state="VIC")
        # 30 improving horses that win
        for i in range(30):
            _make_runner(session, race, tab_no=i+1, position=1,
                         form_history=[
                             {"pos": 2, "date": "2025-06-01"},
                             {"pos": 4, "date": "2025-05-15"},
                             {"pos": 7, "date": "2025-05-01"},
                         ])
        # 30 declining horses that lose
        for i in range(30):
            _make_runner(session, race, tab_no=i+31, position=8,
                         form_history=[
                             {"pos": 8, "date": "2025-06-01"},
                             {"pos": 5, "date": "2025-05-15"},
                             {"pos": 2, "date": "2025-05-01"},
                         ])
        session.commit()

        patterns = analyse_coming_into_form(session)
        improving = [p for p in patterns if "improving" in p.dimension]
        assert len(improving) >= 1
        assert improving[0].win_rate > 0.4


class TestClassMovers:
    def test_downgrade_detected(self, session):
        """Horses dropping in class should show patterns."""
        race = _make_race(session, race_id=1, prize_money=50000, state="VIC")
        # 25 downgrade winners (were in 100k races)
        for i in range(25):
            _make_runner(session, race, tab_no=i+1, position=1,
                         sp=3.0, opening_odds=3.5, prep_runs=2,
                         form_history=[
                             {"pos": 3, "prize_money": 100000},
                             {"pos": 4, "prize_money": 100000},
                             {"pos": 5, "prize_money": 100000},
                         ])
        # 25 downgrade losers
        for i in range(25):
            _make_runner(session, race, tab_no=i+26, position=6,
                         sp=8.0, opening_odds=7.0, prep_runs=1,
                         form_history=[
                             {"pos": 8, "prize_money": 100000},
                             {"pos": 9, "prize_money": 120000},
                         ])
        session.commit()

        patterns = analyse_class_movers(session)
        # Should have downgrade patterns
        downgrades = [p for p in patterns if "downgrade" in p.dimension]
        assert len(downgrades) >= 1

    def test_upgrade_detected(self, session):
        """Horses rising in class should show patterns."""
        race = _make_race(session, race_id=2, prize_money=150000, state="NSW")
        # 20 upgrade winners
        for i in range(20):
            _make_runner(session, race, tab_no=i+1, position=1,
                         sp=5.0, opening_odds=6.0, prep_runs=3,
                         form_history=[
                             {"pos": 1, "prize_money": 80000},
                             {"pos": 2, "prize_money": 75000},
                             {"pos": 1, "prize_money": 70000},
                         ])
        # 30 upgrade losers
        for i in range(30):
            _make_runner(session, race, tab_no=i+21, position=7,
                         sp=15.0, opening_odds=12.0, prep_runs=1,
                         form_history=[
                             {"pos": 5, "prize_money": 60000},
                             {"pos": 6, "prize_money": 55000},
                         ])
        session.commit()

        patterns = analyse_class_movers(session)
        upgrades = [p for p in patterns if "upgrade" in p.dimension]
        assert len(upgrades) >= 1

    def test_indicators_captured(self, session):
        """Winning indicators (competitive, improving, backed, fit) should be tracked."""
        race = _make_race(session, race_id=3, prize_money=50000, state="VIC")
        # Competitive + backed + fit downgrade winners
        for i in range(25):
            _make_runner(session, race, tab_no=i+1, position=1,
                         sp=2.5, opening_odds=4.0, prep_runs=4,
                         form_history=[
                             {"pos": 2, "prize_money": 100000},
                             {"pos": 3, "prize_money": 110000},
                             {"pos": 4, "prize_money": 120000},
                         ])
        # Non-indicator losers
        for i in range(25):
            _make_runner(session, race, tab_no=i+26, position=8,
                         sp=20.0, opening_odds=15.0, prep_runs=0,
                         form_history=[
                             {"pos": 10, "prize_money": 100000},
                         ])
        session.commit()

        patterns = analyse_class_movers(session)
        # Should find multi-indicator patterns
        multi = [p for p in patterns if "multi_indicator" in p.dimension]
        assert len(multi) >= 1


class TestClassTransitions:
    def test_basic_transitions(self, session):
        """Class drop/rise/same should be classified."""
        # Drop: current 50k, recent 100k — 18/30 winners (60% vs ~37% base)
        race1 = _make_race(session, race_id=1, prize_money=50000)
        for i in range(30):
            _make_runner(session, race1, tab_no=i+1,
                         position=1 if i < 18 else 5,
                         form_history=[{"pos": 3, "prize_money": 100000}])
        # Rise: current 150k, recent 50k — low win rate
        race2 = _make_race(session, race_id=2, prize_money=150000)
        for i in range(30):
            _make_runner(session, race2, tab_no=i+1,
                         position=1 if i < 4 else 8,
                         form_history=[{"pos": 1, "prize_money": 50000}])
        session.commit()

        patterns = analyse_class_transitions(session)
        assert len(patterns) >= 1


class TestMarketEfficiency:
    def test_overperforming_sp_range(self, session):
        """If $3-$5 runners win 35% instead of 25%, should detect."""
        race = _make_race(session, race_id=1, state="VIC", location_type="M")
        # $4 runners winning 35%
        for i in range(35):
            _make_runner(session, race, tab_no=i+1, position=1, sp=4.0)
        for i in range(65):
            _make_runner(session, race, tab_no=i+36, position=5, sp=4.0)
        session.commit()

        patterns = analyse_market_efficiency(session)
        # $3-$5 range should show overperformance
        assert any("$3-$5" in p.dimension for p in patterns)


class TestRunAllAnalyses:
    def test_runs_without_error(self, db_path):
        """All 15 analyses should run on empty DB without crashing."""
        init_db(db_path)
        patterns = run_all_analyses(db_path)
        assert isinstance(patterns, list)
        # Empty DB = no significant patterns
        assert len(patterns) == 0


# ──── Import tests ────


class TestImportFormFile:
    def test_imports_runners(self, session, tmp_path):
        """Import a minimal Form JSON file."""
        form_data = [
            {
                "RaceId": 9999,
                "Name": "Test Horse",
                "RunnerId": 12345,
                "TabNo": 1,
                "Barrier": 3,
                "Weight": 58.0,
                "Age": 4,
                "Sex": "Gelding",
                "Country": "AUS",
                "Position": 1,
                "Margin": 0.0,
                "PriceSP": 3.5,
                "CareerStarts": 10,
                "CareerWins": 3,
                "CareerSeconds": 2,
                "CareerThirds": 1,
                "WinPct": 30.0,
                "PlacePct": 60.0,
                "PrizeMoney": 50000,
                "HandicapRating": 72,
                "Last10": "1234567890",
                "PrepRuns": 2,
                "Trainer": {"FullName": "T Jones", "TrainerId": 100},
                "Jockey": {"FullName": "J Smith", "JockeyId": 200},
                "TrackRecord": {"Starts": 3, "Firsts": 1, "Seconds": 1, "Thirds": 0},
                "DistanceRecord": {"Starts": 5, "Firsts": 2, "Seconds": 1, "Thirds": 1},
                "TrackDistRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "FirstUpRecord": {"Starts": 2, "Firsts": 1, "Seconds": 0, "Thirds": 0},
                "SecondUpRecord": {"Starts": 2, "Firsts": 0, "Seconds": 1, "Thirds": 0},
                "GoodRecord": {"Starts": 6, "Firsts": 2, "Seconds": 1, "Thirds": 1},
                "SoftRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "HeavyRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "FirmRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "SyntheticRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "Group1Record": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "Group2Record": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "Group3Record": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "JumpsRecord": {"Starts": 0, "Firsts": 0, "Seconds": 0, "Thirds": 0},
                "JockeyA2E_Career": None,
                "JockeyA2E_Last100": None,
                "TrainerA2E_Career": None,
                "TrainerA2E_Last100": None,
                "TrainerJockeyA2E_Career": None,
                "TrainerJockeyA2E_Last100": None,
                "FormId": 55555,
                "JockeyClaim": 0,
                "OriginalBarrier": 3,
                "EmergencyIndicator": False,
                "Forms": [{
                    "MeetingDate": "2025-05-01T00:00:00",
                    "Track": {"Name": "Caulfield", "State": "VIC"},
                    "Distance": 1400,
                    "RaceClass": "BM78",
                    "TrackCondition": "G4",
                    "Position": 2,
                    "Margin": 1.5,
                    "PriceSP": 4.0,
                    "Starters": 10,
                    "InRun": "settling_down,3;m800,3;m400,2;finish,2;",
                    "Flucs": "opening,3.50;mid,3.80;starting,4.00;",
                    "OfficialRaceTime": "00:01:24.5000000",
                    "PFRaceTime": None,
                    "PrepRuns": 1,
                    "Weight": 57.0,
                    "Barrier": 5,
                    "Rail": None,
                    "TrackConditionNumber": 4,
                    "PrizeMoney": 80000,
                    "AgeRestrictions": None,
                    "SexRestrictions": None,
                    "RaceName": "Test Race",
                    "KRI": 50,
                    "PriceBF": None,
                    "PriceTAB": None,
                    "StewardsReport": None,
                    "GearChanges": None,
                    "HasSectionalData": False,
                    "IsBarrierTrial": False,
                    "Top4Finishers": None,
                    "FormId": 44444,
                    "JockeyClaim": 0,
                    "OriginalBarrier": 5,
                    "EmergencyIndicator": False,
                    "Jockey": {"FullName": "A Rider"},
                    "MeetingDateUTC": None,
                }],
            }
        ]

        filepath = tmp_path / "250615_Flemington.json"
        filepath.write_text(json.dumps(form_data))

        stats = import_form_file(session, filepath, date(2025, 6, 15), "Flemington")
        session.commit()

        assert stats["races"] == 1
        assert stats["runners"] == 1

        # Verify race
        race = session.query(HistoricalRace).first()
        assert race.venue == "Flemington"
        assert race.race_id == 9999

        # Verify runner
        runner = session.query(HistoricalRunner).first()
        assert runner.horse_name == "Test Horse"
        assert runner.won is True
        assert runner.placed is True
        assert runner.barrier == 3
        assert runner.starting_price == 3.5
        assert runner.jockey == "J Smith"
        assert runner.form_history is not None
        history = json.loads(runner.form_history)
        assert len(history) == 1
        assert history[0]["venue"] == "Caulfield"

    def test_idempotent(self, session, tmp_path):
        """Importing the same file twice should not duplicate data."""
        form_data = [{
            "RaceId": 8888,
            "Name": "Double Import",
            "RunnerId": 111,
            "TabNo": 1, "Barrier": 1, "Weight": 56.0,
            "Age": 3, "Sex": "Filly", "Country": "AUS",
            "Position": 3, "Margin": 2.0, "PriceSP": 8.0,
            "CareerStarts": 5, "CareerWins": 1, "CareerSeconds": 1, "CareerThirds": 1,
            "WinPct": 20.0, "PlacePct": 60.0, "PrizeMoney": 20000,
            "HandicapRating": 0, "Last10": "321", "PrepRuns": 1,
            "Trainer": {"FullName": "Trainer"}, "Jockey": {"FullName": "Jockey"},
            "TrackRecord": None, "DistanceRecord": None, "TrackDistRecord": None,
            "FirstUpRecord": None, "SecondUpRecord": None,
            "GoodRecord": None, "SoftRecord": None, "HeavyRecord": None,
            "FirmRecord": None, "SyntheticRecord": None,
            "Group1Record": None, "Group2Record": None, "Group3Record": None,
            "JumpsRecord": None,
            "JockeyA2E_Career": None, "JockeyA2E_Last100": None,
            "TrainerA2E_Career": None, "TrainerA2E_Last100": None,
            "TrainerJockeyA2E_Career": None, "TrainerJockeyA2E_Last100": None,
            "FormId": 777, "JockeyClaim": 0, "OriginalBarrier": 1,
            "EmergencyIndicator": False, "Forms": [],
        }]

        filepath = tmp_path / "250101_Test.json"
        filepath.write_text(json.dumps(form_data))

        import_form_file(session, filepath, date(2025, 1, 1), "Test")
        session.commit()
        import_form_file(session, filepath, date(2025, 1, 1), "Test")
        session.commit()

        assert session.query(HistoricalRunner).count() == 1


class TestImportSectionals:
    def test_imports_sectionals(self, session, tmp_path):
        """Import sectional data and attach to existing race."""
        # First create a race
        race = _make_race(session, race_id=5000)
        runner = _make_runner(session, race, tab_no=1, position=1)
        runner.form_id = 99999
        session.commit()

        sect_data = {
            "payLoad": [{
                "raceId": 5000,
                "raceNo": 1,
                "track": "Flemington",
                "distance": 1600,
                "trackCondition": "G4",
                "timeToFinish": 96.5,
                "timeTo800": 48.2,
                "timeTo600": 59.8,
                "timeTo400": 71.5,
                "timeTo200": 84.1,
                "last600Time": 36.7,
                "last400Time": 25.0,
                "last200Time": 12.4,
                "timeTo1200": None,
                "timeTo1000": None,
                "timeTo100": None,
                "early200Average": None,
                "split1210": None, "split108": None, "split86": None,
                "split64": None, "split42": None, "split21": None,
                "last1200Time": None, "last1000Time": None,
                "last800Time": None, "last100Time": None,
                "officialTime": None, "officialSectionalTime": None,
                "officialSectionalDistance": None,
                "meetingDate": "2025-06-15",
                "meetingDateUTC": None, "meetingId": 1,
                "windDirection": "NW", "windSpeed": 15.0,
                "runnerSectionals": [{
                    "formId": 99999,
                    "tabNumber": 1,
                    "runnerId": 12345,
                    "runnerName": "Test Horse",
                    "timeTo1200": None, "timeTo1000": None,
                    "timeTo800": 48.5, "timeTo600": 60.0,
                    "timeTo400": 72.0, "timeTo200": 84.5,
                    "timeTo100": 90.2, "timeToFin": 96.8,
                    "early200Average": None,
                    "pos1200": None, "pos1000": None,
                    "pos800": 3, "pos600": 2,
                    "pos400": 1, "pos200": 1,
                    "pos100": 1, "posFin": 1,
                    "marg1200": None, "marg1000": None,
                    "marg800": 1.5, "marg600": 0.8,
                    "marg400": 0.0, "marg200": 0.5,
                    "marg100": 0.8, "margFin": 1.2,
                    "split1210": None, "split108": None,
                    "split86": None, "split64": None,
                    "split42": None, "split21": None,
                    "last1200Time": None, "last1000Time": None,
                    "last800Time": None, "last600Time": 36.8,
                    "last400Time": 24.8, "last200Time": 12.3,
                    "last100Time": 6.6,
                    "wides1200": None, "wides1000": None,
                    "wides800": 3, "wides600": 3,
                    "wides400": 4, "wides200": 5,
                    "wides100": None, "widesFin": 6,
                    "meetingRankTo600": None,
                    "meetingRank1210": None, "meetingRank108": None,
                    "meetingRank86": None, "meetingRank64": None,
                    "meetingRank42": None, "meetingRank21": None,
                    "meetingRank12F": None, "meetingRank10F": None,
                    "meetingRank8F": None,
                    "meetingRank6F": 1, "meetingRank4F": 1,
                    "meetingRank2F": 2, "meetingRank1F": 1,
                    "eflaG100": None, "eflaG200": None,
                    "eflaG400": None, "eflaG600": None,
                    "eflaG800": None, "eflaG1000": None, "eflaG1200": None,
                }],
            }]
        }

        filepath = tmp_path / "250615_Flemington.json"
        filepath.write_text(json.dumps(sect_data))

        stats = import_sectionals_file(session, filepath)
        session.commit()

        assert stats["sectionals"] == 1
        assert stats["races_updated"] == 1

        # Verify sectional record
        sect = session.query(HistoricalSectional).first()
        assert sect.pos_800 == 3
        assert sect.pos_fin == 1
        assert sect.last_200 == 12.3
        assert sect.wides_400 == 4

        # Verify race updated
        race = session.query(HistoricalRace).first()
        assert race.time_to_finish == 96.5
        assert race.wind_direction == "NW"
