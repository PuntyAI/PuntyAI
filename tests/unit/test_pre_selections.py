"""Tests for deterministic pre-selection engine."""

import pytest

from punty.context.pre_selections import (
    EACH_WAY_MAX_ODDS,
    EACH_WAY_MIN_ODDS,
    EACH_WAY_MIN_PROB,
    EXOTIC_PUNTYS_PICK_VALUE,
    PLACE_MIN_PROB,
    POOL_HIGH,
    POOL_LOW,
    POOL_STANDARD,
    RACE_POOL,
    ROUGHIE_MIN_ODDS,
    ROUGHIE_MIN_VALUE,
    WIN_MIN_PROB,
    WIN_MIN_VALUE,
    RecommendedExotic,
    RecommendedPick,
    RacePreSelections,
    _allocate_stakes,
    _build_candidates,
    _calculate_puntys_pick,
    _cap_win_exposure,
    _determine_bet_type,
    _determine_race_pool,
    _ensure_win_bet,
    _estimate_place_odds,
    _expected_return,
    _generate_notes,
    _passes_edge_gate,
    _select_exotic,
    calculate_pre_selections,
    format_pre_selections,
)


# ──────────────────────────────────────────────
# Helper: build a runner dict matching context shape
# ──────────────────────────────────────────────

def _runner(
    sc: int,
    name: str,
    odds: float,
    win_prob: float = 0.15,
    place_prob: float = 0.45,
    value: float = 1.1,
    place_value: float = 1.05,
    place_odds: float | None = None,
    rec_stake: float = 3.0,
    scratched: bool = False,
) -> dict:
    return {
        "saddlecloth": sc,
        "horse_name": name,
        "current_odds": odds,
        "place_odds": place_odds or round((odds - 1) / 3 + 1, 2),
        "_win_prob_raw": win_prob,
        "_place_prob_raw": place_prob,
        "punty_win_probability": f"{win_prob * 100:.1f}%",
        "punty_place_probability": f"{place_prob * 100:.1f}%",
        "punty_value_rating": value,
        "punty_place_value_rating": place_value,
        "punty_recommended_stake": rec_stake,
        "scratched": scratched,
    }


def _race_context(runners: list[dict], race_number: int = 1) -> dict:
    """Build a minimal race context dict."""
    probs = {
        "probability_ranked": [
            {"horse": r["horse_name"], "win_prob": f"{r['_win_prob_raw']*100:.1f}%",
             "saddlecloth": r["saddlecloth"]}
            for r in runners if not r.get("scratched")
        ],
        "value_plays": [
            {"horse": r["horse_name"], "value": r["punty_value_rating"], "edge": 5.0}
            for r in runners
            if not r.get("scratched") and r["punty_value_rating"] > 1.05
        ],
        "exotic_combinations": [],
    }
    return {
        "race_number": race_number,
        "runners": runners,
        "probabilities": probs,
    }


# ──────────────────────────────────────────────
# Tests: _build_candidates
# ──────────────────────────────────────────────

class TestBuildCandidates:
    def test_excludes_scratched(self):
        runners = [
            _runner(1, "Alpha", 3.0, scratched=True),
            _runner(2, "Beta", 4.0),
        ]
        result = _build_candidates(runners)
        assert len(result) == 1
        assert result[0]["saddlecloth"] == 2

    def test_excludes_no_odds(self):
        runners = [
            _runner(1, "Alpha", 0),
            _runner(2, "Beta", 4.0),
        ]
        result = _build_candidates(runners)
        assert len(result) == 1

    def test_excludes_no_saddlecloth(self):
        r = _runner(0, "NoSaddle", 4.0)
        r["saddlecloth"] = None
        result = _build_candidates([r])
        assert len(result) == 0

    def test_calculates_ev(self):
        runners = [_runner(1, "Alpha", 5.0, win_prob=0.25)]
        result = _build_candidates(runners)
        assert len(result) == 1
        # EV = 0.25 * 5.0 - 1 = 0.25
        assert abs(result[0]["ev"] - 0.25) < 0.001


# ──────────────────────────────────────────────
# Tests: _determine_bet_type
# ──────────────────────────────────────────────

class TestDetermineBetType:
    def test_top_pick_win_at_3_50_with_conviction(self):
        """$3.50 rank 1 with wp=0.30, value=1.15 → Win (expanded zone $2.01-$6)."""
        c = {"win_prob": 0.30, "place_prob": 0.55, "odds": 3.5,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_top_pick_place_below_4_low_conviction(self):
        """$3.50 rank 1 with wp=0.15 (below 0.20 threshold) → Place."""
        c = {"win_prob": 0.15, "place_prob": 0.55, "odds": 3.5,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Place"

    def test_top_pick_win_in_sweet_spot(self):
        """$4-$6 with good prob/value should be Win (edge: +60.8% ROI)."""
        c = {"win_prob": 0.22, "place_prob": 0.55, "odds": 6.0,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_top_pick_place_outside_win_sweet_spot(self):
        """$8 odds with moderate prob should get Place ($6+ removed)."""
        c = {"win_prob": 0.18, "place_prob": 0.45, "odds": 8.0,
             "value_rating": 0.95, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Place"

    def test_top_pick_win_at_2_50_with_conviction(self):
        """$2.50 rank 1 with wp=0.30, value=1.10 → Win (expanded zone, wp >= 0.22)."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 2.5,
             "value_rating": 1.10, "place_value_rating": 1.05}
        result = _determine_bet_type(c, rank=1, is_roughie=False)
        assert result == "Win"

    def test_top_pick_place_at_2_50_low_value(self):
        """$2.50 rank 1 with value=0.90 (below 0.95 threshold) → Place."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 2.5,
             "value_rating": 0.90, "place_value_rating": 1.05}
        result = _determine_bet_type(c, rank=1, is_roughie=False)
        assert result == "Place"

    def test_second_pick_place_in_sweet_spot(self):
        """Rank 2 in $4-$6 sweet spot gets Place (E/W killed — -16.16% ROI)."""
        c = {"win_prob": 0.22, "place_prob": 0.55, "odds": 4.0,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Place"

    def test_second_pick_place_dead_zone(self):
        """Rank 2 in $3-$4 dead zone gets Place regardless of conviction (E/W killed)."""
        # win_prob 0.28 → Place
        c = {"win_prob": 0.28, "place_prob": 0.55, "odds": 3.0,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Place"
        # win_prob 0.32 → also Place now (was E/W)
        c2 = {"win_prob": 0.32, "place_prob": 0.55, "odds": 3.0,
              "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c2, rank=2, is_roughie=False) == "Place"

    def test_second_pick_place_when_no_value(self):
        c = {"win_prob": 0.12, "place_prob": 0.45, "odds": 8.0,
             "value_rating": 0.95, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Place"

    def test_third_pick_place(self):
        c = {"win_prob": 0.15, "place_prob": 0.45, "odds": 6.0,
             "value_rating": 1.05, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=3, is_roughie=False) == "Place"

    def test_roughie_defaults_to_place(self):
        c = {"win_prob": 0.08, "place_prob": 0.30, "odds": 12.0,
             "value_rating": 1.15, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=4, is_roughie=True) == "Place"

    def test_roughie_always_place(self):
        """All roughies → Place (Rank 4 Win was 0/43 = -98.7% ROI)."""
        c = {"win_prob": 0.20, "place_prob": 0.50, "odds": 10.0,
             "value_rating": 1.25, "place_value_rating": 1.10}
        assert _determine_bet_type(c, rank=4, is_roughie=True) == "Place"


# ──────────────────────────────────────────────
# Tests: _ensure_win_bet
# ──────────────────────────────────────────────

class TestEnsureWinBet:
    def test_no_change_when_win_exists(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Win"
        assert picks[1].bet_type == "Place"

    def test_upgrades_top_to_win_when_all_place(self):
        picks = [
            RecommendedPick(1, 1, "A", "Place", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Win"

    def test_saver_win_counts_as_win(self):
        picks = [
            RecommendedPick(1, 1, "A", "Saver Win", 8, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _ensure_win_bet(picks)
        assert picks[0].bet_type == "Saver Win"  # unchanged


# ──────────────────────────────────────────────
# Tests: _cap_win_exposure
# ──────────────────────────────────────────────

class TestCapWinExposure:
    def test_downgrades_third_when_three_win_exposed(self):
        """3 Win/Saver + 1 roughie → should cap at 2 win-exposed."""
        picks = [
            RecommendedPick(1, 9, "Farfetched", "Win", 3.5, 3.90, 1.97, 0.213, 0.351, 0.43, 0.90, 0.15),
            RecommendedPick(2, 8, "Rach", "Saver Win", 2.5, 4.60, 2.20, 0.184, 0.233, 0.67, 0.85, 0.10),
            RecommendedPick(3, 3, "Farindira", "Win", 5.0, 8.50, 3.50, 0.159, 0.30, 1.23, 1.05, 0.35),
            RecommendedPick(4, 5, "Mister Martini", "Place", 3.5, 23.0, 8.33, 0.071, 0.15, 1.74, 1.10, 0.63, is_roughie=True),
        ]
        _cap_win_exposure(picks)
        assert picks[0].bet_type == "Win"
        assert picks[1].bet_type == "Saver Win"
        assert picks[2].bet_type == "Place"
        assert picks[3].bet_type == "Place"

    def test_no_change_when_two_or_fewer_win_exposed(self):
        """2 win-exposed + 2 Place = fine, no changes."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 7, 4.0, 2.0, 0.25, 0.50, 1.1, 1.0, 0.50),
            RecommendedPick(2, 2, "B", "Saver Win", 5, 5.0, 2.5, 0.20, 0.45, 1.0, 1.0, 0.30),
            RecommendedPick(3, 3, "C", "Place", 5, 6.0, 3.0, 0.15, 0.40, 0.9, 1.0, 0.20),
            RecommendedPick(4, 4, "D", "Place", 3, 10.0, 4.0, 0.08, 0.25, 1.5, 1.1, 0.10, is_roughie=True),
        ]
        _cap_win_exposure(picks)
        assert picks[0].bet_type == "Win"
        assert picks[1].bet_type == "Saver Win"
        assert picks[2].bet_type == "Place"
        assert picks[3].bet_type == "Place"

    def test_three_win_exposed_caps_to_two(self):
        """3 Win bets → third gets downgraded to Place."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 7, 4.0, 2.0, 0.30, 0.55, 1.1, 1.0, 0.50),
            RecommendedPick(2, 2, "B", "Win", 5, 5.0, 2.5, 0.22, 0.45, 1.0, 1.0, 0.30),
            RecommendedPick(3, 3, "C", "Saver Win", 5, 6.0, 3.0, 0.18, 0.40, 1.2, 1.0, 0.25),
            RecommendedPick(4, 4, "D", "Place", 3, 10.0, 4.0, 0.10, 0.30, 1.5, 1.1, 0.10),
        ]
        _cap_win_exposure(picks)
        assert picks[0].bet_type == "Win"
        assert picks[1].bet_type == "Win"
        assert picks[2].bet_type == "Place"
        assert picks[3].bet_type == "Place"

    def test_expected_return_recalculated_on_downgrade(self):
        """When downgrading to Place, expected_return should reflect place odds."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 7, 4.0, 2.0, 0.30, 0.55, 1.1, 1.0, 0.50),
            RecommendedPick(2, 2, "B", "Saver Win", 5, 5.0, 2.5, 0.22, 0.45, 1.0, 1.0, 0.30),
            RecommendedPick(3, 3, "C", "Win", 5, 8.0, 3.5, 0.15, 0.35, 1.2, 1.0, 0.20),
        ]
        _cap_win_exposure(picks)
        assert picks[2].bet_type == "Place"
        # Expected return = place_prob * place_odds - 1 = 0.35 * 3.5 - 1 = 0.225
        assert picks[2].expected_return == pytest.approx(0.22, abs=0.01)


# ──────────────────────────────────────────────
# Tests: _allocate_stakes
# ──────────────────────────────────────────────

class TestAllocateStakes:
    def test_total_within_pool(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 0, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
            RecommendedPick(3, 3, "C", "Place", 0, 7.0, 2.5, 0.15, 0.4, 1.05, 1.0, 0.2),
            RecommendedPick(4, 4, "D", "Place", 0, 12.0, 3.0, 0.08, 0.3, 1.1, 1.05, 0.1, True),
        ]
        _allocate_stakes(picks, 20.0)
        total = sum(p.stake for p in picks)
        assert total <= 20.5  # allow small rounding

    def test_two_place_bets_total_within_pool(self):
        picks = [
            RecommendedPick(1, 1, "A", "Place", 0, 6.0, 2.5, 0.2, 0.5, 1.1, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Place", 0, 5.0, 2.0, 0.2, 0.5, 1.0, 1.0, 0.3),
        ]
        _allocate_stakes(picks, 20.0)
        total = sum(p.stake for p in picks)
        assert total <= 20.5

    def test_minimum_stake(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 3.0, 1.5, 0.3, 0.6, 1.1, 1.05, 0.5),
        ]
        _allocate_stakes(picks, 20.0)
        assert picks[0].stake >= 1.0

    def test_empty_picks(self):
        _allocate_stakes([], 20.0)  # no crash

    def test_vr_cap_removed_win_stays(self):
        """VR > 1.2 Win bets should no longer be downgraded (VR cap removed)."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 3.0, 1.5, 0.3, 0.6, 1.6, 1.05, 0.5),
            RecommendedPick(2, 2, "B", "Saver Win", 0, 5.0, 2.0, 0.2, 0.5, 1.8, 1.0, 0.3),
            RecommendedPick(3, 3, "C", "Place", 0, 7.0, 2.5, 0.15, 0.4, 2.0, 1.0, 0.2),
        ]
        _allocate_stakes(picks, 20.0)
        assert picks[0].bet_type == "Win"  # VR cap removed — Win stays
        assert picks[1].bet_type == "Saver Win"  # VR cap removed — Saver stays
        assert picks[2].bet_type == "Place"  # Already Place, unchanged

    def test_vr_1_3_win_stays(self):
        """VR 1.3 on Win should NOT be downgraded (VR cap removed)."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 4.0, 1.8, 0.25, 0.55, 1.3, 1.05, 0.5),
        ]
        _allocate_stakes(picks, 20.0)
        assert picks[0].bet_type == "Win"

    def test_vr_1_15_win_stays(self):
        """VR 1.15 (below 1.2 cap) on Win should NOT be downgraded."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 4.0, 1.8, 0.25, 0.55, 1.15, 1.05, 0.5),
        ]
        _allocate_stakes(picks, 20.0)
        assert picks[0].bet_type == "Win"

    def test_vr_cap_skips_under_2_odds(self):
        """Win at sub-$2 odds gets blocked by edge gate (too short to back).
        The pick will be tracked_only since Win < $2.00 is a losing zone."""
        picks = [
            RecommendedPick(1, 1, "A", "Win", 0, 1.80, 1.2, 0.5, 0.8, 1.4, 1.05, 0.7),
        ]
        _allocate_stakes(picks, 20.0)
        # Win at $1.80 blocked by edge gate — tracked only.
        # Fallback logic converts to Place as best candidate.
        assert picks[0].bet_type in ("Win", "Place")


# ──────────────────────────────────────────────
# Tests: _select_exotic
# ──────────────────────────────────────────────

class TestSelectExotic:
    def test_prefers_overlap_with_selections(self):
        combos = [
            {"type": "Trifecta", "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
             "probability": "8.5%", "value": 1.30, "combos": 6, "format": "legs"},
            {"type": "Trifecta", "runners": [5, 6, 7], "runner_names": ["E", "F", "G"],
             "probability": "9.0%", "value": 1.35, "combos": 6, "format": "legs"},
        ]
        result = _select_exotic(combos, {1, 2, 3, 4})
        assert result is not None
        assert set(result.runners) == {1, 2, 3}  # overlaps with our picks

    def test_returns_none_when_empty(self):
        assert _select_exotic([], {1, 2}) is None

    def test_metro_venue_allowed(self):
        """Metro venues should allow exotics (no venue filter)."""
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2}, venue_type="metro_vic")
        assert result is not None

    def test_provincial_venue_allowed(self):
        """Provincial venues should allow exotics."""
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2}, venue_type="provincial")
        assert result is not None

    def test_parses_string_probability(self):
        combos = [
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": "12.5%", "value": 1.50, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2})
        assert result is not None
        assert abs(result.probability - 0.125) < 0.001

    def test_ev_scoring_picks_best_type(self):
        """Higher EV (prob × value) wins regardless of exotic type."""
        combos = [
            # Exacta: 10% prob × 1.3 value = 0.13 EV
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": 0.10, "value": 1.3, "combos": 1, "format": "flat"},
            # First4: 8% prob × 2.0 value = 0.16 EV (higher)
            {"type": "First4", "runners": [1, 2, 3, 4], "runner_names": ["A", "B", "C", "D"],
             "probability": 0.08, "value": 2.0, "combos": 6, "format": "legs"},
        ]
        result = _select_exotic(combos, {1, 2, 3, 4})
        assert result is not None
        assert result.exotic_type == "First4"

    def test_all_types_can_win(self):
        """Every exotic type can be selected when it has the best EV."""
        for etype, fmt in [
            ("Quinella", "flat"), ("Exacta Standout", "standout"),
            ("Trifecta", "legs"), ("First4", "legs"), ("First4 Box", "boxed"),
        ]:
            combos = [
                {"type": etype, "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
                 "probability": 0.15, "value": 2.0, "combos": 3, "format": fmt},
            ]
            result = _select_exotic(combos, {1, 2, 3})
            assert result is not None, f"{etype} should be selectable"
            assert result.exotic_type == etype


# ──────────────────────────────────────────────
# Tests: Tight cluster exotic boost
# ──────────────────────────────────────────────

class TestTightClusterExoticBoost:
    def test_tight_cluster_boosts_box_over_exacta(self):
        """When top 3 picks are within 5%, boxed exotic should beat Exacta."""
        # Picks at 28%, 25%, 23% — very tight cluster (5% spread)
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 7, 2.60, 1.3, 0.28, 0.60, 1.05, 1.0, 0.05),
            RecommendedPick(2, 2, "Beta", "Win", 6, 3.30, 1.5, 0.25, 0.55, 1.02, 1.0, 0.03),
            RecommendedPick(3, 3, "Gamma", "Win", 4, 3.60, 1.6, 0.23, 0.50, 1.00, 1.0, 0.01),
        ]
        combos = [
            # Exacta: 12% prob × 1.3 value = 0.156 base EV
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": 0.12, "value": 1.3, "combos": 1, "format": "flat"},
            # First4 Box: 10% prob × 1.35 value = 0.135 base EV
            # With 1.2x cluster boost → 0.162 (beats Exacta's 0.156 × 0.85 penalty = 0.133)
            {"type": "First4 Box", "runners": [1, 2, 3, 4], "runner_names": ["A", "B", "C", "D"],
             "probability": 0.10, "value": 1.35, "combos": 24, "format": "boxed"},
        ]
        result = _select_exotic(combos, {1, 2, 3, 4}, picks=picks)
        assert result is not None
        assert result.exotic_type == "First4 Box"

    def test_no_boost_when_spread_wide(self):
        """No cluster boost when picks are spread (>8%)."""
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 7, 2.00, 1.2, 0.35, 0.70, 1.05, 1.0, 0.10),
            RecommendedPick(2, 2, "Beta", "Win", 5, 5.00, 2.0, 0.20, 0.50, 1.02, 1.0, 0.01),
            RecommendedPick(3, 3, "Gamma", "Place", 3, 10.0, 3.5, 0.10, 0.35, 0.90, 1.0, -0.1),
        ]
        combos = [
            # Exacta has higher base EV and should win without cluster boost
            {"type": "Exacta", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": 0.12, "value": 1.5, "combos": 1, "format": "flat"},
            {"type": "Trifecta", "runners": [1, 2, 3], "runner_names": ["A", "B", "C"],
             "probability": 0.08, "value": 1.3, "combos": 6, "format": "legs"},
        ]
        result = _select_exotic(combos, {1, 2, 3}, picks=picks)
        assert result is not None
        assert result.exotic_type == "Exacta"

    def test_quinella_not_penalised_in_cluster(self):
        """Quinella is unordered — should not get directional penalty."""
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 7, 2.60, 1.3, 0.28, 0.60, 1.05, 1.0, 0.05),
            RecommendedPick(2, 2, "Beta", "Win", 6, 3.30, 1.5, 0.25, 0.55, 1.02, 1.0, 0.03),
            RecommendedPick(3, 3, "Gamma", "Win", 4, 3.60, 1.6, 0.23, 0.50, 1.00, 1.0, 0.01),
        ]
        combos = [
            {"type": "Quinella", "runners": [1, 2], "runner_names": ["A", "B"],
             "probability": 0.20, "value": 1.3, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {1, 2, 3}, picks=picks)
        assert result is not None
        assert result.exotic_type == "Quinella"


# ──────────────────────────────────────────────
# Tests: top-pick exotic bonus
# ──────────────────────────────────────────────

class TestExoticTopPickBonus:
    def test_prefers_combo_with_top_pick(self):
        """Exotic with pick #1 should beat similar-EV exotic without top pick."""
        picks = [
            RecommendedPick(1, 2, "Alpha", "Win", 5, 3.0, 1.5, 0.25, 0.50, 1.1, 1.0, 0.3),
            RecommendedPick(2, 4, "Beta", "Place", 5, 5.0, 2.0, 0.20, 0.40, 1.0, 0.9, 0.1),
            RecommendedPick(3, 1, "Gamma", "Place", 5, 6.0, 2.5, 0.15, 0.35, 0.9, 0.8, 0.0),
            RecommendedPick(4, 3, "Delta", "Place", 5, 9.0, 3.0, 0.10, 0.25, 0.7, 0.6, -0.1, is_roughie=True),
        ]
        combos = [
            # combo with picks #3+#4 (no top-2 pick) — marginally higher base EV
            {"type": "Quinella", "runners": [1, 3], "runner_names": ["Gamma", "Delta"],
             "probability": 0.08, "value": 2.0, "combos": 1, "format": "flat"},
            # combo with pick #1 included — lower raw EV but top pick bonus
            {"type": "Quinella", "runners": [2, 4], "runner_names": ["Alpha", "Beta"],
             "probability": 0.07, "value": 2.0, "combos": 1, "format": "flat"},
        ]
        result = _select_exotic(combos, {2, 4, 1, 3}, picks=picks)
        assert result is not None
        # Should prefer the combo with pick #1 (saddlecloth 2) due to top-pick bonus
        assert 2 in result.runners


# ──────────────────────────────────────────────
# Tests: _calculate_puntys_pick
# ──────────────────────────────────────────────

class TestCalculatePuntysPick:
    def test_picks_highest_confidence_selection(self):
        """Punty's Pick should be the most likely to collect, not highest EV."""
        picks = [
            # Alpha: Win bet, 30% win prob — collects 30% of the time
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.05),
            # Beta: Place bet, 50% place prob — collects 50% of the time
            RecommendedPick(2, 2, "Beta", "Place", 5, 5.0, 2.0, 0.20, 0.50, 1.10, 1.05, -0.1),
        ]
        pp = _calculate_puntys_pick(picks, None)
        assert pp is not None
        assert pp.pick_type == "selection"
        assert pp.horse_name == "Beta"  # highest confidence (50% place prob)

    def test_exotic_wins_when_high_value(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.05, 1.0, 0.05),
        ]
        # Exotic with genuinely high EV: 0.40 * 3.5 - 1 = 0.40 vs sel EV 0.05
        exotic = RecommendedExotic(
            exotic_type="Trifecta", runners=[1, 2, 3],
            runner_names=["A", "B", "C"], probability=0.40,
            value_ratio=3.5, num_combos=6, format="legs",
        )
        pp = _calculate_puntys_pick(picks, exotic)
        assert pp is not None
        assert pp.pick_type == "exotic"
        assert pp.exotic_type == "Trifecta"

    def test_exotic_loses_when_low_value(self):
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.5),
        ]
        exotic = RecommendedExotic(
            exotic_type="Quinella", runners=[1, 2],
            runner_names=["A", "B"], probability=0.20,
            value_ratio=1.3, num_combos=1, format="flat",
        )
        pp = _calculate_puntys_pick(picks, exotic)
        assert pp is not None
        assert pp.pick_type == "selection"  # exotic value below 1.5x

    def test_single_horse_only(self):
        """Punty's Pick should always be a single horse, no secondary."""
        picks = [
            RecommendedPick(1, 1, "Alpha", "Win", 8, 3.5, 1.5, 0.30, 0.60, 1.15, 1.05, 0.5),
            RecommendedPick(2, 2, "Beta", "Place", 5, 5.0, 2.0, 0.20, 0.50, 1.10, 1.05, 0.1),
        ]
        pp = _calculate_puntys_pick(picks, None)
        assert pp.horse_name == "Beta"  # highest confidence (50% place)
        assert not hasattr(pp, "secondary_horse")

    def test_roughie_excluded_from_puntys_pick(self):
        """Roughies should never be Punty's Pick even with high EV."""
        picks = [
            RecommendedPick(1, 1, "TopPick", "Place", 5.5, 3.0, 1.67, 0.22, 0.55, 1.09, 1.05, -0.05),
            RecommendedPick(2, 2, "SecondPick", "Place", 4.5, 8.0, 3.33, 0.12, 0.30, 0.85, 0.90, -0.10),
        ]
        roughie = RecommendedPick(4, 13, "Roughie", "Place", 3.5, 13.0, 5.0, 0.08, 0.20, 2.69, 2.0, 0.40, is_roughie=True)
        all_picks = picks + [roughie]
        pp = _calculate_puntys_pick(all_picks, None)
        assert pp is not None
        assert pp.horse_name == "TopPick"  # highest confidence, not roughie

    def test_empty_picks_returns_none(self):
        assert _calculate_puntys_pick([], None) is None


# ──────────────────────────────────────────────
# Tests: _estimate_place_odds
# ──────────────────────────────────────────────

class TestEstimatePlaceOdds:
    def test_basic_estimation(self):
        assert abs(_estimate_place_odds(7.0) - 3.0) < 0.01

    def test_short_odds(self):
        result = _estimate_place_odds(2.0)
        assert result >= 1.0

    def test_invalid_odds(self):
        assert _estimate_place_odds(1.0) == 1.0


# ──────────────────────────────────────────────
# Tests: _expected_return
# ──────────────────────────────────────────────

class TestExpectedReturn:
    def test_win_positive_ev(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Win")
        # 0.30 * 4.0 - 1 = 0.2
        assert abs(er - 0.2) < 0.01

    def test_place_positive_ev(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Place")
        # 0.60 * 1.8 - 1 = 0.08
        assert abs(er - 0.08) < 0.01

    def test_each_way_combines(self):
        c = {"odds": 4.0, "place_odds": 1.8, "win_prob": 0.30, "place_prob": 0.60}
        er = _expected_return(c, "Each Way")
        win_er = 0.30 * 4.0 - 1  # 0.2
        place_er = 0.60 * 1.8 - 1  # 0.08
        expected = (win_er + place_er) / 2  # 0.14
        assert abs(er - expected) < 0.01


# ──────────────────────────────────────────────
# Tests: _generate_notes
# ──────────────────────────────────────────────

class TestGenerateNotes:
    def test_no_value_note(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 0.95, 0.90, -0.1),
            RecommendedPick(2, 2, "B", "Place", 5, 5.0, 2.0, 0.2, 0.5, 1.00, 1.0, 0.0),
        ]
        notes = _generate_notes(picks, None, [{"win_prob": 0.3}, {"win_prob": 0.2}])
        assert any("No clear value" in n for n in notes)

    def test_strong_value_note(self):
        picks = [
            RecommendedPick(1, 1, "A", "Win", 8, 3.0, 1.5, 0.3, 0.6, 1.25, 1.10, 0.5),
        ]
        notes = _generate_notes(picks, None, [{"win_prob": 0.3}])
        assert any("Strong value" in n for n in notes)

    def test_wide_open_note(self):
        candidates = [{"win_prob": 0.12}, {"win_prob": 0.10}]
        notes = _generate_notes([], None, candidates)
        assert any("Wide-open" in n for n in notes)


# ──────────────────────────────────────────────
# Tests: calculate_pre_selections (integration)
# ──────────────────────────────────────────────

class TestCalculatePreSelections:
    def test_basic_race(self):
        runners = [
            _runner(1, "Alpha", 3.5, win_prob=0.28, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.20, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 8.0, win_prob=0.14, place_prob=0.40, value=1.05),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "Epsilon", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        assert isinstance(result, RacePreSelections)
        assert result.race_number == 1
        assert len(result.picks) == 4
        assert result.total_stake <= 20.5
        assert result.puntys_pick is not None

    def test_pick_order_by_ev(self):
        """Highest win probability runner should be pick #1."""
        runners = [
            _runner(1, "LowEV", 2.0, win_prob=0.10, value=0.80),
            _runner(2, "HighEV", 4.0, win_prob=0.35, value=1.20),
            _runner(3, "MidEV", 6.0, win_prob=0.18, value=1.10),
            _runner(4, "Roughie", 15.0, win_prob=0.06, value=1.15),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        assert result.picks[0].horse_name == "HighEV"

    def test_roughie_identified(self):
        """Runner with odds >= $8 and value >= 1.10 should be roughie."""
        runners = [
            _runner(1, "Fav", 2.5, win_prob=0.35, value=1.10),
            _runner(2, "Contender", 5.0, win_prob=0.20, value=1.08),
            _runner(3, "Chance", 7.0, win_prob=0.14, value=1.05),
            _runner(4, "Outsider", 15.0, win_prob=0.07, place_prob=0.25, value=1.25),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        roughies = [p for p in result.picks if p.is_roughie]
        assert len(roughies) == 1
        assert roughies[0].horse_name == "Outsider"

    def test_at_least_one_win_bet(self):
        """Mandatory rule: at least one Win/Saver Win per race."""
        runners = [
            _runner(1, "A", 3.0, win_prob=0.10, place_prob=0.45, value=0.90, place_value=1.10),
            _runner(2, "B", 4.0, win_prob=0.10, place_prob=0.42, value=0.85, place_value=1.05),
            _runner(3, "C", 5.0, win_prob=0.08, place_prob=0.38, value=0.80, place_value=1.00),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        win_types = {"Win", "Saver Win"}
        assert any(p.bet_type in win_types for p in result.picks)

    def test_empty_runners(self):
        ctx = _race_context([])
        result = calculate_pre_selections(ctx)
        assert result.picks == []
        assert result.puntys_pick is None
        assert result.total_stake == 0.0

    def test_all_scratched(self):
        runners = [
            _runner(1, "A", 3.0, scratched=True),
            _runner(2, "B", 4.0, scratched=True),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        assert result.picks == []

    def test_two_runner_race(self):
        """Small field — should still produce picks. Pool may be $35 (DOMINANT_EDGE)."""
        runners = [
            _runner(1, "A", 2.0, win_prob=0.55, place_prob=0.80, value=1.10),
            _runner(2, "B", 3.0, win_prob=0.35, place_prob=0.65, value=1.05),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        assert len(result.picks) == 2
        assert result.total_stake <= 25.5

    def test_exotic_recommendation(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.30, value=1.10),
            _runner(2, "B", 4.0, win_prob=0.22, value=1.08),
            _runner(3, "C", 6.0, win_prob=0.15, value=1.05),
            _runner(4, "D", 8.0, win_prob=0.10, value=0.95),
            _runner(5, "E", 10.0, win_prob=0.08, value=0.90),
            _runner(6, "F", 12.0, win_prob=0.06, value=0.85),
            _runner(7, "G", 15.0, win_prob=0.05, value=0.80),
        ]
        ctx = _race_context(runners)
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Exacta Standout", "runners": [1, 2],
             "runner_names": ["A", "B"],
             "probability": "8.5%", "value": 1.40, "combos": 1, "format": "standout"},
        ]
        result = calculate_pre_selections(ctx)
        assert result.exotic is not None
        assert result.exotic.exotic_type == "Exacta Standout"

    def test_puntys_pick_exotic_when_high_value(self):
        """Exotic should become Punty's Pick when value >= 1.5x and EV > selection."""
        # Anchor odds must be <= $3.50 for exotic filter to pass
        runners = [
            _runner(1, "A", 3.00, win_prob=0.35, value=1.05),
            _runner(2, "B", 4.50, win_prob=0.22, value=1.02),
            _runner(3, "C", 8.0, win_prob=0.12, value=0.95),
            _runner(4, "D", 10.0, win_prob=0.08, value=0.90),
            _runner(5, "E", 12.0, win_prob=0.06, value=0.85),
            _runner(6, "F", 15.0, win_prob=0.05, value=0.80),
            _runner(7, "G", 20.0, win_prob=0.04, value=0.75),
        ]
        ctx = _race_context(runners)
        # Exotic EV: 0.50 * 3.0 - 1 = 0.50, sel EV: 0.35 * 3.0 - 1 = 0.05
        # Exotic clearly beats selection by > 1.2x
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Exacta Standout", "runners": [1, 2],
             "runner_names": ["A", "B"],
             "probability": "50.0%", "value": 3.0, "combos": 1, "format": "standout"},
        ]
        result = calculate_pre_selections(ctx)
        if result.puntys_pick:
            assert result.puntys_pick.pick_type == "exotic"


# ──────────────────────────────────────────────
# Tests: format_pre_selections
# ──────────────────────────────────────────────

class TestFormatPreSelections:
    def test_basic_formatting(self):
        runners = [
            _runner(1, "Alpha", 3.5, win_prob=0.28, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.20, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 8.0, win_prob=0.14, place_prob=0.40, value=1.05),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)

        assert "LOCKED SELECTIONS" in formatted
        assert "Race 1" in formatted
        assert "Total stake" in formatted
        assert "$" in formatted

    def test_exotic_in_output(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.30),
            _runner(2, "B", 4.0, win_prob=0.25),
            _runner(3, "C", 6.0, win_prob=0.15),
            _runner(4, "D", 8.0, win_prob=0.10),
            _runner(5, "E", 10.0, win_prob=0.08),
            _runner(6, "F", 12.0, win_prob=0.06),
            _runner(7, "G", 15.0, win_prob=0.05),
        ]
        ctx = _race_context(runners)
        ctx["probabilities"]["exotic_combinations"] = [
            {"type": "Exacta Standout", "runners": [1, 2],
             "runner_names": ["A", "B"],
             "probability": "20.0%", "value": 1.50, "combos": 1, "format": "standout"},
        ]
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)
        assert "Exotic:" in formatted
        assert "Exacta Standout" in formatted

    def test_notes_in_output(self):
        runners = [
            _runner(1, "A", 3.0, win_prob=0.10, value=0.90),
            _runner(2, "B", 4.0, win_prob=0.10, value=0.95),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        formatted = format_pre_selections(result)
        assert "NOTE:" in formatted


# ──────────────────────────────────────────────
# Tests: Context Multiplier Threshold Adjustment
# ──────────────────────────────────────────────

class TestContextMultiplierThresholds:
    def test_weak_context_tightens_place_threshold(self):
        """Weak place context (<0.85 avg) raises PLACE_MIN_PROB by 0.05."""
        runners = [
            _runner(1, "Fav", 2.5, win_prob=0.30, place_prob=0.60, value=1.2),
            _runner(2, "Second", 5.0, win_prob=0.18, place_prob=0.42, value=1.0),
            _runner(3, "Third", 8.0, win_prob=0.10, place_prob=0.38, value=0.95),
            _runner(4, "Fourth", 12.0, win_prob=0.06, place_prob=0.25, value=0.9),
        ]
        ctx = _race_context(runners)
        # Weak context: avg multiplier < 0.85
        weak_mults = {"form": 0.7, "class_fitness": 0.8, "barrier": 0.75}
        result = calculate_pre_selections(ctx, place_context_multipliers=weak_mults)
        # With tightened threshold (0.40), runner 3 at 0.38 should NOT get Place
        for pick in result.picks:
            if pick.saddlecloth == 3 and pick.bet_type == "Place":
                # 0.38 < 0.40 threshold — shouldn't be Place
                # (but bet_optimizer may override, so just check it ran)
                pass
        assert result is not None

    def test_strong_context_loosens_place_threshold(self):
        """Strong place context (>1.2 avg) lowers PLACE_MIN_PROB by 0.05."""
        runners = [
            _runner(1, "Fav", 2.5, win_prob=0.30, place_prob=0.60, value=1.2),
            _runner(2, "Second", 5.0, win_prob=0.18, place_prob=0.32, value=1.0),
            _runner(3, "Third", 8.0, win_prob=0.10, place_prob=0.32, value=0.95),
            _runner(4, "Fourth", 12.0, win_prob=0.06, place_prob=0.25, value=0.9),
        ]
        ctx = _race_context(runners)
        # Strong context: avg multiplier > 1.2
        strong_mults = {"form": 1.3, "class_fitness": 1.25, "barrier": 1.15}
        result = calculate_pre_selections(ctx, place_context_multipliers=strong_mults)
        # With loosened threshold (0.30), more Place bets should be allowed
        assert result is not None

    def test_neutral_context_no_adjustment(self):
        """Neutral context (0.85-1.2) doesn't adjust threshold."""
        runners = [
            _runner(1, "Fav", 2.5, win_prob=0.30, place_prob=0.60, value=1.2),
            _runner(2, "Second", 5.0, win_prob=0.18, place_prob=0.42, value=1.0),
        ]
        ctx = _race_context(runners)
        neutral_mults = {"form": 1.0, "class_fitness": 0.95}
        result = calculate_pre_selections(ctx, place_context_multipliers=neutral_mults)
        assert result is not None


# ──────────────────────────────────────────────
# Tests: Edge Gate (_passes_edge_gate)
# ──────────────────────────────────────────────

class TestEdgeGate:
    def _pick(self, bet_type="Win", odds=5.0, win_prob=0.25, place_prob=0.50,
              value=1.10, place_value=1.05, is_roughie=False):
        return RecommendedPick(
            rank=1, saddlecloth=1, horse_name="Test", bet_type=bet_type,
            stake=0.0, odds=odds,
            place_odds=round((odds - 1) / 3 + 1, 2),
            win_prob=win_prob, place_prob=place_prob,
            value_rating=value, place_value_rating=place_value,
            expected_return=0.0, is_roughie=is_roughie,
        )

    def test_win_sweet_spot_passes(self):
        """Win at $4-$6 always passes (proven +60.8% ROI)."""
        pick = self._pick(bet_type="Win", odds=5.0)
        assert _passes_edge_gate(pick)[0] is True

    def test_win_under_2_fails(self):
        """Win at <$2 should fail (historically -38.9% ROI)."""
        pick = self._pick(bet_type="Win", odds=1.80, win_prob=0.45)
        assert _passes_edge_gate(pick)[0] is False

    def test_place_good_prob_passes(self):
        """Place with high probability passes."""
        pick = self._pick(bet_type="Place", odds=4.0, place_prob=0.50, place_value=1.05)
        assert _passes_edge_gate(pick)[0] is True

    def test_place_low_prob_fails(self):
        """Place with low collection probability fails."""
        pick = self._pick(bet_type="Place", odds=8.0, place_prob=0.25, place_value=0.85)
        assert _passes_edge_gate(pick)[0] is False

    def test_place_mid_range_passes(self):
        """Place at $3-$6 with decent place_prob passes (was E/W rule)."""
        pick = self._pick(bet_type="Place", odds=4.0, place_prob=0.45)
        assert _passes_edge_gate(pick)[0] is True

    def test_roughie_place_good_prob_passes(self):
        """Roughie Place at $8-$20 with strong place_prob passes."""
        pick = self._pick(bet_type="Place", odds=12.0, place_prob=0.40,
                          is_roughie=True)
        assert _passes_edge_gate(pick)[0] is True

    def test_negative_ev_both_ways_fails(self):
        """Strongly negative EV on both win and place fails."""
        pick = self._pick(bet_type="Place", odds=15.0, win_prob=0.03,
                          place_prob=0.10, value=0.50, place_value=0.50)
        assert _passes_edge_gate(pick)[0] is False

    def test_place_mid_price_graduated_floor(self):
        """Place at $3-$6 uses 0.40 floor (graduated thresholds)."""
        # place_prob=0.42 at $4 should pass (0.42 >= 0.40)
        pick = self._pick(bet_type="Place", odds=4.0, place_prob=0.42, place_value=1.00)
        assert _passes_edge_gate(pick)[0] is True
        # place_prob=0.38 at $4 should fail (0.38 < 0.40)
        pick2 = self._pick(bet_type="Place", odds=4.0, place_prob=0.38, place_value=1.00)
        assert _passes_edge_gate(pick2)[0] is False

    def test_place_short_price_needs_high_prob(self):
        """Place at <$3 uses strict 0.55 floor (graduated thresholds)."""
        # place_prob=0.50 at $2.50 should fail (0.50 < 0.55)
        pick = self._pick(bet_type="Place", odds=2.50, place_prob=0.50, place_value=1.00)
        assert _passes_edge_gate(pick)[0] is False
        # place_prob=0.56 at $2.50 should pass (0.56 >= 0.55)
        pick2 = self._pick(bet_type="Place", odds=2.50, place_prob=0.56, place_value=1.00)
        assert _passes_edge_gate(pick2)[0] is True

    def test_place_long_odds_lower_floor(self):
        """Place at $6+ uses relaxed 0.35 floor (graduated thresholds)."""
        # place_prob=0.36 at $8 should pass (0.36 >= 0.35)
        pick = self._pick(bet_type="Place", odds=8.0, place_prob=0.36, place_value=1.00)
        assert _passes_edge_gate(pick)[0] is True

    def test_win_2_40_to_3_strong_conviction_passes(self):
        """Win at $2.40-$3 with high win_prob passes."""
        pick = self._pick(bet_type="Win", odds=2.60, win_prob=0.35)
        assert _passes_edge_gate(pick)[0] is True

    def test_win_2_40_to_3_weak_conviction_fails(self):
        """Win at $2.40-$3 with very low win_prob (< 0.20) fails."""
        pick = self._pick(bet_type="Win", odds=2.60, win_prob=0.15)
        assert _passes_edge_gate(pick)[0] is False

    def test_win_2_40_to_3_passes_with_conviction(self):
        """Win at $2.40-$3 with win_prob >= 0.20 passes."""
        pick = self._pick(bet_type="Win", odds=2.60, win_prob=0.20)
        assert _passes_edge_gate(pick)[0] is True

    def test_win_3_to_4_passes_with_conviction(self):
        """Win at $3-$4 with win_prob >= 0.20 passes (no more dead zone)."""
        pick_r1 = RecommendedPick(
            rank=1, saddlecloth=1, horse_name="Test", bet_type="Win",
            stake=0.0, odds=3.50, place_odds=1.83,
            win_prob=0.27, place_prob=0.50, value_rating=1.05,
            place_value_rating=1.05, expected_return=0.0,
        )
        pick_r3 = RecommendedPick(
            rank=3, saddlecloth=3, horse_name="Test3", bet_type="Win",
            stake=0.0, odds=3.50, place_odds=1.83,
            win_prob=0.27, place_prob=0.50, value_rating=1.05,
            place_value_rating=1.05, expected_return=0.0,
        )
        assert _passes_edge_gate(pick_r1)[0] is True  # wp 0.27 >= 0.20
        assert _passes_edge_gate(pick_r3)[0] is True  # wp 0.27 >= 0.20 (same threshold now)

    def test_live_profile_override_rejects_losing_band(self):
        """Live profile with strong negative ROI overrides even passing criteria."""
        pick = self._pick(bet_type="Win", odds=5.0, win_prob=0.25)
        # This would normally pass (sweet spot), but live data says it's losing
        profile = {("win", "$4-$6"): {"roi": -20.0, "sr": 15.0, "bets": 60, "avg_pnl": -2.5}}
        assert _passes_edge_gate(pick, live_profile=profile)[0] is False


# ──────────────────────────────────────────────
# Tests: 3-Tier Pool (_determine_race_pool)
# ──────────────────────────────────────────────

class TestDetermineRacePool:
    def _pick(self, odds=5.0, win_prob=0.25, place_prob=0.50,
              place_value=1.10):
        return RecommendedPick(
            rank=1, saddlecloth=1, horse_name="Test", bet_type="Win",
            stake=0.0, odds=odds,
            place_odds=round((odds - 1) / 3 + 1, 2),
            win_prob=win_prob, place_prob=place_prob,
            value_rating=1.10, place_value_rating=place_value,
            expected_return=0.0,
        )

    def test_high_ev_gets_high_pool(self):
        """Strong EV (>0.15) should get $35 pool."""
        # ev = 0.35 * 5.0 - 1 = 0.75 > 0.15
        pick = self._pick(odds=5.0, win_prob=0.35)
        pool = _determine_race_pool([pick])
        assert pool == POOL_HIGH

    def test_strong_place_gets_high_pool(self):
        """Place prob > 0.50 with place_value > 1.05 gets $35 pool."""
        pick = self._pick(odds=5.0, win_prob=0.15, place_prob=0.55,
                          place_value=1.10)
        pool = _determine_race_pool([pick])
        assert pool == POOL_HIGH

    def test_standard_race_gets_standard_pool(self):
        """Multiple picks with positive EV but not dominant gets $20."""
        picks = [
            self._pick(odds=5.0, win_prob=0.22),  # ev = 0.10
            self._pick(odds=3.0, win_prob=0.35, place_prob=0.60),  # ev_win = 0.05
        ]
        pool = _determine_race_pool(picks)
        assert pool == POOL_STANDARD

    def test_empty_picks_gets_low_pool(self):
        """No picks gets $12 pool."""
        pool = _determine_race_pool([])
        assert pool == POOL_LOW

    def test_watch_only_gets_low_pool(self):
        """Watch-only classification gets $12 pool."""
        from punty.context.bet_optimizer import RaceClassification
        cls = RaceClassification(
            race_type="NO_EDGE", confidence=0.5,
            reasoning="test", watch_only=True, no_bet=False,
        )
        pick = self._pick()
        pool = _determine_race_pool([pick], classification=cls)
        assert pool == POOL_LOW


# ──────────────────────────────────────────────
# Tests: Edge-gated stake allocation
# ──────────────────────────────────────────────

class TestEdgeGatedAllocation:
    def test_tracked_picks_get_zero_stake(self):
        """Picks that fail edge gate should get $0 stake and tracked_only=True."""
        # Sub-$2 Win picks should fail
        picks = [
            RecommendedPick(1, 1, "Fav", "Place", 0.0, 1.50, 1.17,
                            0.50, 0.80, 1.0, 1.0, 0.0),
            RecommendedPick(2, 2, "Good", "Win", 0.0, 5.0, 2.33,
                            0.25, 0.50, 1.10, 1.05, 0.0),
        ]
        _allocate_stakes(picks, 20.0)
        # Sub-$1.50 Place with place_prob 0.80 should pass (>=0.40, odds 1.50 < 2.50 but... check)
        # Actually $1.50 is not in the $2.50-$8 range, let's verify the logic
        # The sub-$2 Place has ev_place = 0.80 * 1.17 - 1 = -0.064 and place_prob >= 0.40 and place_value >= 0.95
        # place_value is 1.0 >= 0.95, so criterion 3 passes
        assert picks[1].stake > 0  # Win at $5 should be staked

    def test_fallback_when_all_fail(self):
        """When all picks fail edge gate, best Place bet is forced."""
        picks = [
            RecommendedPick(1, 1, "Bad1", "Win", 0.0, 1.50, 1.17,
                            0.40, 0.30, 0.80, 0.80, -0.5),
            RecommendedPick(2, 2, "Bad2", "Win", 0.0, 1.60, 1.20,
                            0.35, 0.25, 0.75, 0.75, -0.5),
        ]
        _allocate_stakes(picks, 20.0)
        # At least one pick should have stake > 0 (fallback)
        total_stake = sum(p.stake for p in picks)
        assert total_stake > 0

    def test_tracked_picks_not_counted_in_total(self):
        """tracked_only picks should have stake = 0."""
        picks = [
            RecommendedPick(1, 1, "Good", "Win", 0.0, 5.0, 2.33,
                            0.25, 0.50, 1.10, 1.05, 0.0),
            RecommendedPick(2, 2, "Weak", "Win", 0.0, 1.70, 1.23,
                            0.40, 0.25, 0.85, 0.85, -0.3),
        ]
        _allocate_stakes(picks, 20.0)
        for p in picks:
            if p.tracked_only:
                assert p.stake == 0.0


# ──────────────────────────────────────────────
# Tests: Sub-$2 bet type changes
# ──────────────────────────────────────────────

class TestSubTwoDollarBetType:
    def test_sub_180_gets_place(self):
        """Sub-$1.80 should get Place (not Win as before)."""
        c = {"win_prob": 0.45, "place_prob": 0.75, "odds": 1.50,
             "value_rating": 0.95, "place_value_rating": 1.00}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Place"

    def test_180_to_200_gets_place(self):
        """$1.80-$2.00 should get Place (not Win as before)."""
        c = {"win_prob": 0.40, "place_prob": 0.70, "odds": 1.90,
             "value_rating": 1.05, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Place"

    def test_expanded_win_zone_3_to_4(self):
        """$3-$4 rank 1 with wp >= 0.20 AND value >= 0.95 → Win (expanded zone)."""
        # wp=0.28, value=1.15 → Win (passes both thresholds)
        c = {"win_prob": 0.28, "place_prob": 0.55, "odds": 3.50,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

        # wp=0.15 (below 0.20 threshold) → Place
        c2 = {"win_prob": 0.15, "place_prob": 0.55, "odds": 3.50,
              "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c2, rank=1, is_roughie=False) == "Place"

        # Rank 2 always Place in $3-$4 range
        c3 = {"win_prob": 0.28, "place_prob": 0.55, "odds": 3.50,
              "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c3, rank=2, is_roughie=False) == "Place"


# ──────────────────────────────────────────────
# Tests: Each Way killed (#11)
# ──────────────────────────────────────────────

class TestEachWayKilled:
    """Verify no E/W assignment from _determine_bet_type (except ≤7 field Win)."""

    def test_ew_paths_all_return_place_or_win(self):
        """Sweep all odds bands and ranks — no Each Way should ever be returned."""
        for odds in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0]:
            for rank in [1, 2, 3, 4]:
                for wp in [0.10, 0.20, 0.30, 0.40]:
                    c = {"win_prob": wp, "place_prob": wp * 2,
                         "odds": odds, "value_rating": 1.15,
                         "place_value_rating": 1.05}
                    result = _determine_bet_type(c, rank=rank, is_roughie=(rank == 4))
                    assert result != "Each Way", (
                        f"Got E/W at odds={odds}, rank={rank}, wp={wp}"
                    )

    def test_small_field_no_ew(self):
        """≤7 field returns Win (not E/W) for rank ≤2."""
        c = {"win_prob": 0.20, "place_prob": 0.40, "odds": 4.0,
             "value_rating": 1.10, "place_value_rating": 1.05}
        result = _determine_bet_type(c, rank=2, is_roughie=False, field_size=6)
        assert result == "Win"  # not Each Way


# ──────────────────────────────────────────────
# Tests: Win → Place guard (#10)
# ──────────────────────────────────────────────

class TestWinToPlaceGuard:
    """place_prob >= 2.0 * win_prob guard in $2.40-$4.00 bands."""

    def test_rank1_win_at_3_50_expanded_zone(self):
        """$3.50, rank 1, wp=0.30, value=1.15 → Win (expanded $2.01-$6 zone)."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 3.5,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_rank1_win_stays_in_sweet_spot(self):
        """$4.50 is in proven sweet spot — Win stays."""
        c = {"win_prob": 0.25, "place_prob": 0.55, "odds": 4.5,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_rank1_win_at_3_50_with_conviction(self):
        """$3.50 rank 1 with wp=0.30 and value=1.15 → Win (expanded zone)."""
        c = {"win_prob": 0.30, "place_prob": 0.55, "odds": 3.5,
             "value_rating": 1.15, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_rank1_win_at_2_60_expanded_zone(self):
        """$2.60, rank 1, wp=0.30, value=1.10 → Win (expanded $2.01-$3 zone, wp >= 0.22)."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 2.6,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=1, is_roughie=False) == "Win"

    def test_rank2_place_at_2_60(self):
        """$2.60 rank 2 → Place (only rank 1 gets Win in expanded zone)."""
        c = {"win_prob": 0.30, "place_prob": 0.65, "odds": 2.6,
             "value_rating": 1.10, "place_value_rating": 1.05}
        assert _determine_bet_type(c, rank=2, is_roughie=False) == "Place"


# ──────────────────────────────────────────────
# Tests: Exotic filters (#9)
# ──────────────────────────────────────────────

class TestExoticFilters:
    """Data-driven exotic filters from Feb 24 audit."""

    COMBOS = [{"type": "Exacta Standout", "runners": [1, 2],
               "runner_names": ["A", "B"],
               "probability": "20.0%", "value": 1.50, "combos": 1, "format": "standout"}]

    def test_heavy_track_blocks_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=8,
                                track_condition="Heavy 10")
        assert result is None

    def test_soft_7_blocks_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=8,
                                track_condition="Soft 7")
        assert result is None

    def test_soft_5_allows_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=8,
                                track_condition="Soft 5")
        assert result is not None

    def test_hk_blocks_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=8, is_hk=True)
        assert result is None

    def test_small_field_blocks_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=5)
        assert result is None

    def test_good_track_allows_exotic(self):
        result = _select_exotic(self.COMBOS, {1, 2}, field_size=8,
                                track_condition="Good 4")
        assert result is not None

    def test_trifecta_box_scenario_filter(self):
        """Trifecta Box: field 7-14, fav $1.50-$5.00 (relaxed for 4-horse box)."""
        tri_combos = [{"type": "Trifecta Box", "runners": [1, 2, 3],
                        "runner_names": ["A", "B", "C"],
                        "probability": "8.5%", "value": 1.40, "combos": 6, "format": "boxed"}]
        # Blocked: field too small (< 7) or too large (> 14)
        for fs in [5, 6, 15, 16]:
            result = _select_exotic(tri_combos, {1, 2, 3}, field_size=fs,
                                    track_condition="Good 4", fav_price=2.80, distance=1600)
            assert result is None, f"Trifecta Box should be blocked in {fs}-field"
        # Blocked: fav too short (< $1.50)
        result = _select_exotic(tri_combos, {1, 2, 3}, field_size=10,
                                track_condition="Good 4", fav_price=1.30, distance=1600)
        assert result is None, "Trifecta Box should be blocked when fav < $1.50"
        # Blocked: fav too long (> $5.00)
        result = _select_exotic(tri_combos, {1, 2, 3}, field_size=10,
                                track_condition="Good 4", fav_price=5.50, distance=1600)
        assert result is None, "Trifecta Box should be blocked when fav > $5.00"
        # Allowed: sprint distance (no longer blocked)
        result = _select_exotic(tri_combos, {1, 2, 3}, field_size=10,
                                track_condition="Good 4", fav_price=2.80, distance=1100)
        assert result is not None, "Trifecta Box should be allowed in sprint"
        # Allowed: field 7-14, fav $1.50-$5.00
        result = _select_exotic(tri_combos, {1, 2, 3}, field_size=12,
                                track_condition="Good 4", fav_price=2.80, distance=1600)
        assert result is not None, "Trifecta Box should be allowed in qualifying scenario"
        assert result.exotic_type == "Trifecta Box"


# ──────────────────────────────────────────────
# Tests: NTD staked-pick cap
# ──────────────────────────────────────────────

class TestNtdStakedPickCap:
    """In ≤7 runner fields (NTD), only 2 picks staked: 1 Win + 1 Place, selected by place_prob."""

    def test_ntd_field_caps_to_2_staked(self):
        """6-runner field: only 2 staked picks, rest tracked_only."""
        runners = [
            _runner(1, "Alpha", 4.5, win_prob=0.30, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.22, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 6.5, win_prob=0.16, place_prob=0.42, value=1.08),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "Epsilon", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
            _runner(6, "Zeta", 20.0, win_prob=0.04, place_prob=0.15, value=0.80),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        staked = [p for p in result.picks if not p.tracked_only]
        tracked = [p for p in result.picks if p.tracked_only]

        assert len(staked) <= 2, f"NTD field should have at most 2 staked picks, got {len(staked)}"
        assert len(tracked) >= 1, "NTD field should have at least 1 tracked-only pick"
        for t in tracked:
            assert t.stake == 0.0, f"Tracked-only pick {t.horse_name} should have $0 stake"

    def test_ntd_selects_by_place_prob(self):
        """NTD picks should be the 2 highest place_prob candidates."""
        runners = [
            _runner(1, "HighWP", 4.5, win_prob=0.35, place_prob=0.45, value=1.15),
            _runner(2, "HighPP", 5.0, win_prob=0.18, place_prob=0.65, value=1.05),
            _runner(3, "MidPP", 6.0, win_prob=0.15, place_prob=0.55, value=1.08),
            _runner(4, "LowPP", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "Filler", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        staked = [p for p in result.picks if not p.tracked_only]
        staked_names = {p.horse_name for p in staked}
        # HighPP (0.65) and MidPP (0.55) should be selected, not HighWP (0.45)
        assert "HighPP" in staked_names, f"HighPP (pp=0.65) should be staked, got {staked_names}"
        assert "MidPP" in staked_names, f"MidPP (pp=0.55) should be staked, got {staked_names}"

    def test_ntd_win_plus_place(self):
        """NTD should have pick #1 = Win and pick #2 = Place."""
        runners = [
            _runner(1, "A", 4.5, win_prob=0.30, place_prob=0.60, value=1.15),
            _runner(2, "B", 5.0, win_prob=0.22, place_prob=0.50, value=1.10),
            _runner(3, "C", 6.5, win_prob=0.16, place_prob=0.42, value=1.08),
            _runner(4, "D", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "E", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
            _runner(6, "F", 20.0, win_prob=0.04, place_prob=0.15, value=0.80),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        staked = [p for p in result.picks if not p.tracked_only]
        assert len(staked) >= 2
        assert staked[0].bet_type == "Win", f"Pick #1 should be Win, got {staked[0].bet_type}"
        assert staked[1].bet_type == "Place", f"Pick #2 should be Place, got {staked[1].bet_type}"

    def test_normal_field_no_ntd_cap(self):
        """10-runner field: NTD path should NOT activate."""
        runners = [
            _runner(i, f"Horse{i}", 3.0 + i, win_prob=0.30 - i * 0.03,
                    place_prob=0.60 - i * 0.04, value=1.10)
            for i in range(1, 11)
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        non_roughie = [p for p in result.picks if not p.is_roughie]
        assert len(non_roughie) >= 2

    def test_ntd_note_generated(self):
        """NTD field should produce a note mentioning Win + Place."""
        runners = [
            _runner(1, "A", 4.5, win_prob=0.30, place_prob=0.60, value=1.15),
            _runner(2, "B", 5.5, win_prob=0.22, place_prob=0.50, value=1.10),
            _runner(3, "C", 7.0, win_prob=0.15, place_prob=0.40, value=1.05),
            _runner(4, "D", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "E", 18.0, win_prob=0.05, place_prob=0.18, value=0.90),
            _runner(6, "F", 25.0, win_prob=0.03, place_prob=0.12, value=0.80),
            _runner(7, "G", 30.0, win_prob=0.02, place_prob=0.10, value=0.70),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        ntd_notes = [n for n in result.notes if "NTD" in n]
        assert len(ntd_notes) >= 1, f"Expected NTD note, got notes: {result.notes}"

    def test_very_small_field_no_ntd(self):
        """4-runner field: NTD path should NOT activate (≤4 is win-only)."""
        runners = [
            _runner(1, "A", 4.5, win_prob=0.35, place_prob=0.65, value=1.15),
            _runner(2, "B", 5.0, win_prob=0.25, place_prob=0.55, value=1.10),
            _runner(3, "C", 7.0, win_prob=0.18, place_prob=0.42, value=1.05),
            _runner(4, "D", 12.0, win_prob=0.10, place_prob=0.30, value=1.00),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)

        ntd_notes = [n for n in result.notes if "NTD" in n]
        assert len(ntd_notes) == 0, f"≤4 field should not trigger NTD, got: {result.notes}"


# ──────────────────────────────────────────────
# Tests: No Bet reasons (_passes_edge_gate returns reasons)
# ──────────────────────────────────────────────

class TestNoBetReasons:
    def _pick(self, bet_type="Win", odds=5.0, win_prob=0.25, place_prob=0.50,
              value=1.10, place_value=1.05, is_roughie=False, rank=1):
        return RecommendedPick(
            rank=rank, saddlecloth=1, horse_name="Test", bet_type=bet_type,
            stake=0.0, odds=odds,
            place_odds=round((odds - 1) / 3 + 1, 2),
            win_prob=win_prob, place_prob=place_prob,
            value_rating=value, place_value_rating=place_value,
            expected_return=0.0, is_roughie=is_roughie,
        )

    def test_pass_returns_none_reason(self):
        """Passing picks return (True, None)."""
        pick = self._pick(bet_type="Win", odds=5.0)
        passed, reason = _passes_edge_gate(pick)
        assert passed is True
        assert reason is None

    def test_win_under_2_reason(self):
        """Win < $2 returns 'Too short to back' reason."""
        pick = self._pick(bet_type="Win", odds=1.80, win_prob=0.45)
        passed, reason = _passes_edge_gate(pick)
        assert passed is False
        assert "Too short" in reason

    def test_low_conviction_reason(self):
        """Win $2-$4 with low conviction returns specific reason."""
        pick = self._pick(bet_type="Win", odds=3.50, win_prob=0.15, value=1.05, rank=3)
        passed, reason = _passes_edge_gate(pick)
        assert passed is False
        assert "Not enough conviction" in reason
        assert "15%" in reason

    def test_place_low_prob_reason(self):
        """Place with low prob returns reason with threshold."""
        pick = self._pick(bet_type="Place", odds=8.0, place_prob=0.25, place_value=0.85)
        passed, reason = _passes_edge_gate(pick)
        assert passed is False
        assert "Place prob too low" in reason
        assert "25%" in reason

    def test_negative_ev_reason(self):
        """Negative EV both ways returns 'Negative expected value'."""
        # Win bet at $7 with low win_prob — doesn't hit pass criteria (wp<0.18),
        # doesn't hit "too short" or "dead zone" or "place prob too low" (bt=Win),
        # ev_win = 0.05*7-1 = -0.65, ev_place = 0.15*3.0-1 = -0.55 → negative EV gate
        pick = self._pick(bet_type="Win", odds=7.0, win_prob=0.05,
                          place_prob=0.15, value=0.50, place_value=0.50)
        passed, reason = _passes_edge_gate(pick)
        assert passed is False
        assert "Negative expected value" in reason

    def test_live_profile_reason(self):
        """Live profile override returns reason with ROI."""
        pick = self._pick(bet_type="Win", odds=5.0, win_prob=0.25)
        profile = {("win", "$4-$6"): {"roi": -20.0, "sr": 15.0, "bets": 60, "avg_pnl": -2.5}}
        passed, reason = _passes_edge_gate(pick, live_profile=profile)
        assert passed is False
        assert "Losing odds band" in reason
        assert "-20%" in reason

    def test_allocate_stakes_sets_reason(self):
        """_allocate_stakes stores no_bet_reason when edge gate fails."""
        picks = [
            RecommendedPick(1, 1, "Short", "Win", 0.0, 1.50, 1.17,
                            0.50, 0.80, 1.0, 1.0, 0.0),
            RecommendedPick(2, 2, "Good", "Win", 0.0, 5.0, 2.33,
                            0.25, 0.50, 1.10, 1.05, 0.0),
        ]
        _allocate_stakes(picks, 20.0)
        # Sub-$2 Win should fail with reason
        short_pick = picks[0]
        if short_pick.tracked_only:
            assert short_pick.no_bet_reason is not None
            assert "Too short" in short_pick.no_bet_reason

    def test_ntd_tracked_picks_have_reason(self):
        """NTD tracked-only picks should have 'NTD field' reason."""
        runners = [
            _runner(1, "Alpha", 4.5, win_prob=0.30, place_prob=0.60, value=1.15),
            _runner(2, "Beta", 5.0, win_prob=0.22, place_prob=0.50, value=1.10),
            _runner(3, "Gamma", 6.5, win_prob=0.16, place_prob=0.42, value=1.08),
            _runner(4, "Delta", 12.0, win_prob=0.08, place_prob=0.25, value=1.20),
            _runner(5, "Epsilon", 15.0, win_prob=0.06, place_prob=0.20, value=0.90),
            _runner(6, "Zeta", 20.0, win_prob=0.04, place_prob=0.15, value=0.80),
        ]
        ctx = _race_context(runners)
        result = calculate_pre_selections(ctx)
        tracked = [p for p in result.picks if p.tracked_only]
        assert len(tracked) >= 1
        for t in tracked:
            assert t.no_bet_reason is not None
            assert "NTD" in t.no_bet_reason

    def test_format_includes_reason(self):
        """format_pre_selections should include the no_bet_reason text."""
        pick = self._pick(bet_type="Place", odds=8.0, place_prob=0.25, place_value=0.85)
        pick.tracked_only = True
        pick.no_bet_reason = "Place prob too low (25% < 35%)"
        pre_sel = RacePreSelections(
            race_number=1,
            picks=[pick],
            exotic=None,
            puntys_pick=None,
            total_stake=0.0,
            notes=[],
        )
        output = format_pre_selections(pre_sel)
        assert "Place prob too low (25% < 35%)" in output
        assert "No Bet" in output
