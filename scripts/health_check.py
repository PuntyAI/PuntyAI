"""Pre-deploy health check — verifies all critical paths work.

Run after every deploy: python scripts/health_check.py
Exit code 0 = all good, 1 = failures found.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

errors = []


def check(name):
    """Decorator for health checks."""
    def wrapper(fn):
        async def run():
            print(f"  {name}...", end=" ", flush=True)
            try:
                result = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                if result:
                    print(f"OK ({result})")
                else:
                    print("OK")
            except Exception as e:
                msg = f"{name}: {e}"
                print(f"FAIL — {e}")
                errors.append(msg)
        return run
    return wrapper


@check("Probability engine returns results")
def test_probability():
    from punty.probability import calculate_race_probabilities

    class MockRunner:
        def __init__(self, id, name, sc, odds):
            self.id = id
            self.horse_name = name
            self.saddlecloth = sc
            self.current_odds = odds
            self.opening_odds = odds
            self.barrier = sc
            self.weight = 57.0
            self.form = "x1x23"
            self.last_five = "12345"
            self.jockey = "J. Smith"
            self.trainer = "T. Jones"
            self.horse_age = 4
            self.horse_sex = "Gelding"
            self.scratched = False
            self.speed_map_position = "midfield"
            self.form_history = "[]"
            self.career_record = "10: 3-2-1"
            self.jockey_stats = None
            self.trainer_stats = None
            self.track_dist_stats = None
            self.distance_stats = None
            self.good_track_stats = None
            self.soft_track_stats = None
            self.heavy_track_stats = None
            self.class_stats = None
            self.handicap_rating = None
            self.days_since_last_run = 14
            self.pf_settle = None
            self.pf_speed_rank = None
            self.pf_map_factor = None
            self.pf_jockey_factor = None
            self.kash_rated_price = None
            self.kash_speed_cat = None
            self.pf_ai_price = None
            self.pf_ai_score = None
            self.pf_assessed_price = None
            self.sire = None
            self.dam_sire = None

    class MockRace:
        def __init__(self):
            self.id = "test-r1"
            self.race_number = 1
            self.distance = 1200
            self.class_ = "Maiden"
            self.name = "Test Race"

    class MockMeeting:
        def __init__(self):
            self.id = "test-meeting"
            self.venue = "Flemington"
            self.track_condition = "Good 4"
            self.rail_position = None
            self.date = "2026-03-27"

    runners = [
        MockRunner("test-r1-1-horse-a", "Horse A", 1, 3.50),
        MockRunner("test-r1-2-horse-b", "Horse B", 2, 5.00),
        MockRunner("test-r1-3-horse-c", "Horse C", 3, 8.00),
        MockRunner("test-r1-4-horse-d", "Horse D", 4, 12.00),
    ]

    probs = calculate_race_probabilities(runners, MockRace(), MockMeeting())
    assert probs, "Empty probs dict"
    assert len(probs) >= 2, f"Only {len(probs)} probs returned"
    return f"{len(probs)} runners scored"


@check("Prob keys match runner.id (THE BUG)")
def test_prob_key_format():
    from punty.probability import calculate_race_probabilities

    class R:
        def __init__(self, id, name, sc):
            self.id = id
            self.horse_name = name
            self.saddlecloth = sc
            for attr in ["current_odds", "opening_odds", "barrier", "weight", "form",
                         "last_five", "jockey", "trainer", "horse_age", "horse_sex",
                         "scratched", "speed_map_position", "form_history", "career_record",
                         "jockey_stats", "trainer_stats", "track_dist_stats", "distance_stats",
                         "good_track_stats", "soft_track_stats", "heavy_track_stats",
                         "class_stats", "handicap_rating", "days_since_last_run",
                         "pf_settle", "pf_speed_rank", "pf_map_factor", "pf_jockey_factor",
                         "kash_rated_price", "kash_speed_cat", "pf_ai_price", "pf_ai_score",
                         "pf_assessed_price", "sire", "dam_sire"]:
                setattr(self, attr, None)
            self.current_odds = 3.0 + sc
            self.barrier = sc
            self.weight = 57.0
            self.horse_age = 4
            self.scratched = False
            self.last_five = "12345"
            self.form_history = "[]"

    class Race:
        id = "test-r1"
        race_number = 1
        distance = 1400
        class_ = "BM68"
        name = "Test"

    class Meeting:
        id = "test"
        venue = "Caulfield"
        track_condition = "Good 4"
        rail_position = None
        date = "2026-03-27"

    runners = [
        R("test-r1-1-alpha", "Alpha", 1),
        R("test-r1-2-beta", "Beta", 2),
        R("test-r1-3-gamma", "Gamma", 3),
        R("test-r1-4-delta", "Delta", 4),
    ]

    probs = calculate_race_probabilities(runners, Race(), Meeting())

    # THE CRITICAL CHECK: runner.id must be a valid key
    matched = 0
    for runner in runners:
        p = probs.get(runner.id)
        if p and p.win_probability > 0:
            matched += 1

    assert matched > 0, (
        f"probs.get(runner.id) returned None for ALL runners. "
        f"Keys: {list(probs.keys())[:3]}, IDs: {[r.id for r in runners[:3]]}"
    )
    return f"{matched}/{len(runners)} matched by runner.id"


@check("Sense check runs")
def test_sense_check():
    from punty.sense_check import sense_check_race
    runners = [
        {"saddlecloth": 1, "current_odds": 3.0, "kash_rated_price": 3.5,
         "pf_ai_score": 80, "scratched": False},
        {"saddlecloth": 2, "current_odds": 5.0, "kash_rated_price": 4.0,
         "pf_ai_score": 60, "scratched": False},
    ]
    result = sense_check_race(1, runners)
    assert result["consensus"] in ("HIGH", "MEDIUM", "LOW")
    return result["consensus"]


@check("Kelly staking calculates")
def test_kelly():
    from punty.betting.queue import calculate_kelly_stake
    stake = calculate_kelly_stake(balance=100, place_probability=0.70, odds=1.50, max_fraction=0.06)
    assert stake >= 0, f"Negative stake: {stake}"
    return f"${stake:.2f} on $100 balance"


@check("Consensus override function")
def test_consensus():
    from punty.sense_check import find_consensus_pick
    runners = [
        {"saddlecloth": 1, "current_odds": 2.0, "kash_rated_price": 2.5,
         "pf_ai_score": 90, "scratched": False},
        {"saddlecloth": 2, "current_odds": 5.0, "kash_rated_price": 6.0,
         "pf_ai_score": 40, "scratched": False},
    ]
    picks = [
        {"saddlecloth": 2, "tip_rank": 1, "horse_name": "Horse B"},
        {"saddlecloth": 1, "tip_rank": 2, "horse_name": "Horse A"},
    ]
    result = find_consensus_pick(picks, runners)
    # Horse A (#1) is fav for all 3 models, and is our R2
    assert result is not None, "Should find consensus on R2"
    assert result["saddlecloth"] == 1
    return f"R{result['tip_rank']} override found"


@check("Speed benchmarks loaded")
async def test_speed():
    from punty.probability import _SPEED_BENCHMARKS
    n = len(_SPEED_BENCHMARKS)
    if n == 0:
        # Try loading from DB
        from punty.models.database import async_session
        from punty.models.settings import AppSettings
        from sqlalchemy import select
        import json
        async with async_session() as db:
            r = await db.execute(select(AppSettings).where(AppSettings.key == "speed_benchmarks"))
            s = r.scalar_one_or_none()
            if s and s.value:
                return f"In DB ({len(json.loads(s.value))} entries) but not loaded in this process"
        raise Exception("No speed benchmarks in DB or memory")
    return f"{n} entries"


@check("J/T context data loaded")
async def test_jt():
    from punty.probability import _JT_CONTEXT_DATA
    if not _JT_CONTEXT_DATA:
        from punty.models.database import async_session
        from punty.models.settings import AppSettings
        from sqlalchemy import select
        import json
        async with async_session() as db:
            r = await db.execute(select(AppSettings).where(AppSettings.key == "jt_context_data"))
            s = r.scalar_one_or_none()
            if s and s.value:
                data = json.loads(s.value)
                total = sum(len(v) for v in data.values() if isinstance(v, dict))
                return f"In DB ({total} entries) but not loaded in this process"
        raise Exception("No J/T context in DB or memory")
    total = sum(len(v) for v in _JT_CONTEXT_DATA.values() if isinstance(v, dict))
    return f"{total} entries"


async def main():
    print("=" * 50)
    print("PuntyAI Health Check")
    print("=" * 50)
    print()

    checks = [
        test_probability,
        test_prob_key_format,
        test_sense_check,
        test_kelly,
        test_consensus,
        test_speed,
        test_jt,
    ]

    for check_fn in checks:
        await check_fn()

    print()
    if errors:
        print(f"FAILED: {len(errors)} check(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
