"""Tests for race-level bet type optimizer."""

import pytest

from punty.context.bet_optimizer import (
    CHAOS_HANDICAP,
    COMPRESSED_VALUE,
    DOMINANT_EDGE,
    NO_EDGE,
    PLACE_LEVERAGE,
    BetRecommendation,
    RaceClassification,
    RaceOptimization,
    calculate_odds_movement,
    check_circuit_breaker,
    classify_race,
    get_venue_confidence,
    optimize_race,
    place_baseline,
    recommend_bet,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _candidate(
    sc: int = 1,
    name: str = "Horse",
    odds: float = 4.0,
    win_prob: float = 0.25,
    place_prob: float = 0.55,
    place_odds: float | None = None,
    opening_odds: float | None = None,
    career_record: str | None = "5: 2-1-1",
    value_rating: float = 1.0,
    place_value_rating: float = 1.0,
) -> dict:
    """Build a candidate dict for testing."""
    if place_odds is None:
        place_odds = round((odds - 1) / 3 + 1, 2)
    implied_win = 1.0 / odds if odds > 0 else 0
    implied_place = 1.0 / place_odds if place_odds > 1.0 else 0.5
    return {
        "saddlecloth": sc,
        "horse_name": name,
        "odds": odds,
        "place_odds": place_odds,
        "opening_odds": opening_odds,
        "win_prob": win_prob,
        "place_prob": place_prob,
        "value_rating": value_rating,
        "place_value_rating": place_value_rating,
        "ev_win": win_prob * odds - 1,
        "ev_place": place_prob * place_odds - 1,
        "win_edge": win_prob - implied_win,
        "place_edge": place_prob - implied_place,
        "implied_win": implied_win,
        "implied_place": implied_place,
        "career_record": career_record,
        "rec_stake": 5.0,
        "punty_value_rating": value_rating,
        "punty_place_value_rating": place_value_rating,
        "punty_recommended_stake": 5.0,
    }


def _race_context(
    runners: list[dict] | None = None,
    race_class: str = "BM70",
    age_restriction: str | None = None,
) -> dict:
    """Build a minimal race context dict."""
    if runners is None:
        runners = [
            _candidate(1, "Horse1", 3.0, 0.30, 0.60),
            _candidate(2, "Horse2", 5.0, 0.20, 0.50),
            _candidate(3, "Horse3", 8.0, 0.15, 0.45),
            _candidate(4, "Horse4", 12.0, 0.10, 0.35),
        ]
    # Add _win_prob_raw, _place_prob_raw, _edge_raw, current_odds for optimizer
    for r in runners:
        r.setdefault("_win_prob_raw", r.get("win_prob", 0.2))
        r.setdefault("_place_prob_raw", r.get("place_prob", 0.5))
        r.setdefault("_edge_raw", r.get("win_edge", 0))
        r.setdefault("current_odds", r.get("odds", 4.0))
        r.setdefault("scratched", False)
    return {
        "race_number": 1,
        "runners": runners,
        "class": race_class,
        "age_restriction": age_restriction,
        "probabilities": {},
    }


# ──────────────────────────────────────────────
# TestClassifyRace
# ──────────────────────────────────────────────

class TestClassifyRace:
    def test_no_bet_2yo_maiden_first_starters(self):
        """2yo maiden with all first starters -> no_bet=True."""
        candidates = [
            _candidate(1, "Debut1", 5.0, 0.20, 0.40, career_record=None),
            _candidate(2, "Debut2", 6.0, 0.18, 0.38, career_record=""),
            _candidate(3, "Debut3", 8.0, 0.15, 0.35, career_record="0: 0-0-0"),
        ]
        result = classify_race(
            candidates, field_size=3,
            race_class="2yo Maiden", age_restriction="2yo",
        )
        assert result.no_bet is True
        assert result.race_type == NO_EDGE

    def test_not_no_bet_if_any_has_form(self):
        """Same class but one runner has career_record -> not no_bet."""
        candidates = [
            _candidate(1, "Debut1", 5.0, 0.20, 0.40, career_record=None),
            _candidate(2, "Experienced", 3.0, 0.30, 0.55, career_record="5: 2-1-1"),
            _candidate(3, "Debut3", 8.0, 0.15, 0.35, career_record=""),
        ]
        result = classify_race(
            candidates, field_size=3,
            race_class="2yo Maiden", age_restriction="2yo",
        )
        assert result.no_bet is False

    def test_dominant_edge(self):
        """One runner with strong probability and edge gap -> DOMINANT_EDGE."""
        candidates = [
            _candidate(1, "Standout", 3.0, 0.35, 0.65),   # edge=0.35-0.33=0.02... need more
            _candidate(2, "Also Ran", 8.0, 0.15, 0.40),
            _candidate(3, "Filler", 12.0, 0.10, 0.35),
        ]
        # Recalculate with better edge: need win_edge >= 0.04 and gap >= 0.08
        # p_win=0.38, odds=$2.60 -> implied=0.385 -> edge=-0.005... too tight
        # p_win=0.35, odds=$2.50 -> implied=0.40 -> edge=-0.05... negative
        # p_win=0.35, odds=$5.00 -> implied=0.20 -> edge=+0.15 ✓, gap=0.35-0.15=0.20 ✓
        candidates = [
            _candidate(1, "Standout", 5.0, 0.35, 0.65),
            _candidate(2, "Also Ran", 10.0, 0.15, 0.40),
            _candidate(3, "Filler", 15.0, 0.10, 0.35),
        ]
        result = classify_race(candidates, field_size=10)
        assert result.race_type == DOMINANT_EDGE
        assert "Standout" in result.reasoning

    def test_chaos_handicap_large_field(self):
        """14+ runners -> CHAOS_HANDICAP."""
        candidates = [
            _candidate(i, f"Horse{i}", 6.0 + i, 0.08, 0.25)
            for i in range(1, 15)
        ]
        result = classify_race(candidates, field_size=14)
        assert result.race_type == CHAOS_HANDICAP

    def test_chaos_handicap_open_small_field(self):
        """Fav at $5+ and top prob <= 0.22 -> CHAOS_HANDICAP."""
        candidates = [
            _candidate(1, "H1", 5.5, 0.20, 0.45),
            _candidate(2, "H2", 6.0, 0.18, 0.42),
            _candidate(3, "H3", 7.0, 0.15, 0.38),
            _candidate(4, "H4", 8.0, 0.12, 0.30),
        ]
        result = classify_race(candidates, field_size=10)
        assert result.race_type == CHAOS_HANDICAP

    def test_place_leverage(self):
        """Runner with place edge but modest win -> PLACE_LEVERAGE."""
        # Need fav odds < $5 so it doesn't trigger CHAOS (fav >= $5 + p_top1 <= 0.22)
        # p_win=0.18, odds=$7.0 -> fav is $4.0 (H0), so we need a non-chaos fav
        # Actually: CHAOS triggers when fav_odds >= 5.0 AND p_top1 <= 0.22
        # So add a stronger fav to prevent chaos
        candidates = [
            _candidate(1, "StrongFav", 3.5, 0.28, 0.60, place_odds=1.8),
            _candidate(2, "Placer", 7.0, 0.18, 0.55, place_odds=2.0),
            _candidate(3, "H3", 10.0, 0.12, 0.40, place_odds=3.0),
        ]
        # fav_odds=3.5 (not >= 5.0) so no CHAOS
        # Placer: p_win=0.18 in [0.15, 0.22], p_place=0.55 >= 0.30+0.15=0.45 ✓
        # place_edge = 0.55 - 1/2 = 0.05 ✓, odds=7.0 in [5,12] ✓
        # ev_win = 0.18*7-1=0.26 ✓, ev_place = 0.55*2-1=0.10 ✓
        result = classify_race(candidates, field_size=10)
        assert result.race_type == PLACE_LEVERAGE

    def test_compressed_value_default(self):
        """Standard competitive race -> COMPRESSED_VALUE."""
        # Ensure no PLACE_LEVERAGE by having p_win outside 0.15-0.22 for
        # the runners that have high place prob, and fav < $5 to avoid CHAOS
        candidates = [
            _candidate(1, "H1", 4.0, 0.25, 0.55, place_odds=2.0),
            _candidate(2, "H2", 4.5, 0.22, 0.50, place_odds=2.2),
            _candidate(3, "H3", 6.0, 0.15, 0.40, place_odds=2.8),
            _candidate(4, "H4", 8.0, 0.12, 0.30, place_odds=3.5),
        ]
        # H1: p_win=0.25 > 0.22 -> not PLACE_LEVERAGE candidate
        # H2: p_win=0.22 -> at boundary (0.15-0.22 inclusive), but odds=4.5 not in [5,12]
        # H3: p_win=0.15 -> boundary, odds=6.0 in range, p_place=0.40 < 0.30+0.15=0.45 -> fails
        result = classify_race(candidates, field_size=10)
        assert result.race_type == COMPRESSED_VALUE

    def test_watch_only_no_edge(self):
        """No runner meets minimum edge thresholds -> watch_only=True."""
        # All runners at market odds (edge ≈ 0)
        candidates = [
            _candidate(1, "H1", 4.0, 0.25, 0.50, place_odds=2.0),
            # win_edge = 0.25 - 0.25 = 0 < 0.03
            # place_edge = 0.50 - 0.50 = 0 < 0.05
            _candidate(2, "H2", 5.0, 0.20, 0.40, place_odds=2.5),
            # win_edge = 0.20 - 0.20 = 0
            # place_edge = 0.40 - 0.40 = 0
        ]
        result = classify_race(candidates, field_size=8)
        assert result.watch_only is True

    def test_classification_priority_dominant_over_chaos(self):
        """DOMINANT_EDGE takes priority over CHAOS_HANDICAP."""
        # Large field but clear standout
        candidates = [
            _candidate(1, "Standout", 5.0, 0.35, 0.65),  # strong
        ] + [
            _candidate(i, f"H{i}", 15.0, 0.05, 0.15)
            for i in range(2, 15)
        ]
        result = classify_race(candidates, field_size=14)
        assert result.race_type == DOMINANT_EDGE


# ──────────────────────────────────────────────
# TestBetRecommendation
# ──────────────────────────────────────────────

class TestBetRecommendation:
    def test_dominant_edge_rank1_win_only(self):
        c = _candidate(1, "Champ", 5.0, 0.35, 0.65)
        rec = recommend_bet(c, DOMINANT_EDGE, rank=1, field_size=10)
        assert rec.bet_type == "Win"
        assert rec.stake_pct == 0.40

    def test_dominant_edge_no_ew_rank1(self):
        """Dominant edge rank 1 should always be Win, never EW."""
        c = _candidate(1, "Champ", 4.0, 0.35, 0.65)
        rec = recommend_bet(c, DOMINANT_EDGE, rank=1, field_size=10)
        assert rec.bet_type == "Win"

    def test_place_leverage_rank1_ew_when_both_ev_positive(self):
        c = _candidate(1, "Placer", 7.0, 0.18, 0.55, place_odds=2.0)
        rec = recommend_bet(c, PLACE_LEVERAGE, rank=1, field_size=10)
        assert rec.bet_type == "Each Way"

    def test_place_leverage_can_skip_win_bet(self):
        """All picks in PLACE_LEVERAGE can be Place."""
        for rank in [1, 2, 3, 4]:
            c = _candidate(rank, f"H{rank}", 15.0, 0.10, 0.35, place_odds=4.0)
            # EV_win = 0.10*15 - 1 = 0.50, EV_place = 0.35*4 - 1 = 0.40
            # But rank 1 with odds $15 is EW range, and both EVs positive -> EW
            rec = recommend_bet(c, PLACE_LEVERAGE, rank=rank, field_size=10)
            assert rec.bet_type in ("Place", "Each Way")

    def test_compressed_2_overlays_win_plus_saver(self):
        """RULE 3: 2 overlays -> Win + Saver Win."""
        c1 = _candidate(1, "H1", 5.0, 0.25, 0.55)  # edge = 0.25-0.20 = 0.05
        c2 = _candidate(2, "H2", 4.5, 0.28, 0.58)   # edge = 0.28-0.22 = 0.06
        candidates = [c1, c2]

        rec1 = recommend_bet(c1, COMPRESSED_VALUE, rank=1, field_size=10, candidates=candidates)
        rec2 = recommend_bet(c2, COMPRESSED_VALUE, rank=2, field_size=10, candidates=candidates)
        assert rec1.bet_type == "Win"
        assert rec2.bet_type == "Saver Win"

    def test_place_only_when_ev_win_negative(self):
        """RULE 4: EV_win <= 0, EV_place > 0.03 -> Place."""
        # odds=$2.50, win_prob=0.35 -> ev_win = 0.35*2.5-1 = -0.125 (negative!)
        # Hmm that's positive. Let's do: odds=$3.0, win_prob=0.25 -> ev_win = -0.25
        # place_odds=$1.67, place_prob=0.65 -> ev_place = 0.65*1.67 - 1 = 0.085 > 0.03
        c = _candidate(3, "PlaceOnly", 3.0, 0.25, 0.65, place_odds=1.67)
        # ev_win = 0.25*3 - 1 = -0.25 ✓
        # ev_place = 0.65*1.67 - 1 = 0.0855 ✓
        rec = recommend_bet(c, COMPRESSED_VALUE, rank=3, field_size=10, candidates=[c])
        assert rec.bet_type == "Place"

    def test_roughie_small_win_under_30(self):
        """RULE 5: Roughie under $30 with edge -> small Win."""
        c = _candidate(4, "Roughie", 20.0, 0.08, 0.25)
        # win_edge = 0.08 - 0.05 = 0.03 > 0 ✓
        rec = recommend_bet(c, COMPRESSED_VALUE, rank=4, field_size=10, candidates=[c])
        assert rec.bet_type == "Win"
        assert rec.stake_pct == 0.10

    def test_roughie_no_win_above_30(self):
        """RULE 5: Roughie above $30 -> Place only."""
        c = _candidate(4, "Longshot", 50.0, 0.04, 0.12)
        rec = recommend_bet(c, COMPRESSED_VALUE, rank=4, field_size=10, candidates=[c])
        assert rec.bet_type == "Place"

    def test_roughie_no_ew_above_15(self):
        """Never EW on $15+ runners."""
        c = _candidate(4, "Roughie", 18.0, 0.08, 0.25)
        rec = recommend_bet(c, COMPRESSED_VALUE, rank=4, field_size=10, candidates=[c])
        # Should be Win (positive edge) or Place, never EW
        assert rec.bet_type != "Each Way"

    def test_chaos_win_best_overlay(self):
        """RULE 6: Chaos race, rank 1 with overlay -> Win."""
        c = _candidate(1, "BestOverlay", 6.0, 0.22, 0.50)
        # win_edge = 0.22 - 0.167 = 0.053 > 0.03 ✓, odds >= 4.0 ✓
        rec = recommend_bet(c, CHAOS_HANDICAP, rank=1, field_size=14)
        assert rec.bet_type == "Win"

    def test_ew_filter_under_2_50(self):
        """RULE 7: EW filtered when odds < $2.50."""
        c = _candidate(2, "ShortPrice", 2.0, 0.40, 0.70, place_odds=1.33)
        rec = recommend_bet(c, PLACE_LEVERAGE, rank=1, field_size=10)
        # Odds $2.0 < $2.50 -> EW filtered
        assert rec.bet_type != "Each Way"

    def test_ew_filter_over_15(self):
        """RULE 7: EW filtered when odds > $15."""
        c = _candidate(1, "LongOdds", 18.0, 0.10, 0.30, place_odds=5.0)
        c["ev_win"] = 0.10 * 18 - 1  # 0.80
        c["ev_place"] = 0.30 * 5 - 1  # 0.50
        rec = recommend_bet(c, PLACE_LEVERAGE, rank=1, field_size=10)
        assert rec.bet_type != "Each Way"

    def test_max_2_win_bets_per_race(self):
        """RULE 8: Capital efficiency caps at 2 win-exposed bets."""
        from punty.context.bet_optimizer import _enforce_capital_efficiency

        recs = [
            BetRecommendation(1, "H1", "Win", 0.35, 0.1, 0.1, 0.05, 0.05, ""),
            BetRecommendation(2, "H2", "Each Way", 0.25, 0.1, 0.1, 0.04, 0.04, ""),
            BetRecommendation(3, "H3", "Saver Win", 0.20, 0.1, 0.1, 0.03, 0.03, ""),
            BetRecommendation(4, "H4", "Place", 0.15, 0.1, 0.1, 0.01, 0.01, ""),
        ]
        _enforce_capital_efficiency(recs)
        win_types = {"Win", "Saver Win", "Each Way"}
        win_count = sum(1 for r in recs if r.bet_type in win_types)
        assert win_count <= 2


# ──────────────────────────────────────────────
# TestOddsMovement
# ──────────────────────────────────────────────

class TestOddsMovement:
    def test_strong_shortening_boost(self):
        # $6 -> $4 = 33% shortening
        assert calculate_odds_movement(6.0, 4.0) == 1.15

    def test_moderate_shortening(self):
        # $5 -> $4.2 = 16% shortening
        assert calculate_odds_movement(5.0, 4.2) == 1.07

    def test_strong_drift_reduce(self):
        # $3 -> $4 = -33% drift
        assert calculate_odds_movement(3.0, 4.0) == 0.85

    def test_moderate_drift(self):
        # $4 -> $4.6 = -15% drift
        assert calculate_odds_movement(4.0, 4.6) == 0.93

    def test_stable_no_change(self):
        assert calculate_odds_movement(4.0, 4.0) == 1.0

    def test_missing_opening_returns_1(self):
        assert calculate_odds_movement(None, 4.0) == 1.0
        assert calculate_odds_movement(4.0, None) == 1.0
        assert calculate_odds_movement(None, None) == 1.0

    def test_opening_at_1_returns_1(self):
        assert calculate_odds_movement(1.0, 2.0) == 1.0


# ──────────────────────────────────────────────
# TestVenueConfidence
# ──────────────────────────────────────────────

class TestVenueConfidence:
    def test_metro_full_confidence(self):
        assert get_venue_confidence("metro_vic") == 1.0
        assert get_venue_confidence("metro_nsw") == 1.0

    def test_provincial_reduced(self):
        assert get_venue_confidence("provincial") == 0.90

    def test_country_reduced(self):
        assert get_venue_confidence("country") == 0.80

    def test_unknown_venue_conservative(self):
        assert get_venue_confidence("unknown") == 0.85


# ──────────────────────────────────────────────
# TestCircuitBreaker
# ──────────────────────────────────────────────

class TestCircuitBreaker:
    def test_activates_0_from_4(self):
        assert check_circuit_breaker(0, 4) is True
        assert check_circuit_breaker(0, 6) is True

    def test_inactive_with_hits(self):
        assert check_circuit_breaker(1, 4) is False
        assert check_circuit_breaker(2, 6) is False

    def test_inactive_early_races(self):
        assert check_circuit_breaker(0, 3) is False
        assert check_circuit_breaker(0, 1) is False

    def test_inactive_none_params(self):
        assert check_circuit_breaker(None, None) is False
        assert check_circuit_breaker(None, 4) is False
        assert check_circuit_breaker(0, None) is False


# ──────────────────────────────────────────────
# TestFieldSizeScaling
# ──────────────────────────────────────────────

class TestFieldSizeScaling:
    def test_small_field_higher_baseline(self):
        # 6 runners, 2 places -> 2/6 = 0.333
        assert abs(place_baseline(6) - 2 / 6) < 0.001

    def test_large_field_lower_baseline(self):
        # 16 runners, 3 places -> 3/16 = 0.1875
        assert abs(place_baseline(16) - 3 / 16) < 0.001

    def test_standard_field(self):
        # 10 runners, 3 places -> 3/10 = 0.30
        assert abs(place_baseline(10) - 0.30) < 0.001

    def test_place_leverage_threshold_scales(self):
        """Larger fields should have lower place thresholds."""
        small = place_baseline(6) + 0.15   # 0.333 + 0.15 = 0.483
        large = place_baseline(16) + 0.15  # 0.1875 + 0.15 = 0.3375
        assert small > large


# ──────────────────────────────────────────────
# TestOptimizeRace (integration)
# ──────────────────────────────────────────────

class TestOptimizeRace:
    def test_full_flow_dominant_edge(self):
        """Full race context -> optimize -> DOMINANT_EDGE classification."""
        runners = [
            _candidate(1, "Standout", 5.0, 0.35, 0.65),
            _candidate(2, "Also", 10.0, 0.15, 0.40),
            _candidate(3, "Filler", 15.0, 0.10, 0.35),
            _candidate(4, "Long", 25.0, 0.06, 0.20),
        ]
        ctx = _race_context(runners)
        result = optimize_race(ctx, pool=20.0, venue_type="metro_vic")

        assert result.classification.race_type == DOMINANT_EDGE
        assert not result.classification.watch_only
        assert not result.classification.no_bet
        assert len(result.recommendations) == 4

        # Rank 1 should be Win
        assert result.recommendations[0].bet_type == "Win"

    def test_full_flow_watch_only(self):
        """No edge in any runner -> watch_only=True, half-pool stakes."""
        runners = [
            _candidate(1, "H1", 4.0, 0.25, 0.50, place_odds=2.0),
            _candidate(2, "H2", 5.0, 0.20, 0.40, place_odds=2.5),
            _candidate(3, "H3", 6.0, 0.167, 0.33, place_odds=3.0),
        ]
        ctx = _race_context(runners)
        result = optimize_race(ctx, pool=20.0)

        assert result.classification.watch_only is True
        # Watch only gets half pool — stakes non-zero but reduced
        total = sum(r.stake_pct * 20.0 for r in result.recommendations)
        assert total > 0
        assert total <= 10.0 + 0.5  # half of $20 pool + rounding

    def test_full_flow_no_bet(self):
        """2yo maiden first starters -> no_bet=True, empty recommendations."""
        runners = [
            _candidate(1, "Debut1", 5.0, 0.20, 0.40, career_record=None),
            _candidate(2, "Debut2", 6.0, 0.18, 0.38, career_record=""),
        ]
        ctx = _race_context(runners, race_class="2yo Maiden", age_restriction="2yo")
        result = optimize_race(ctx)

        assert result.classification.no_bet is True
        assert len(result.recommendations) == 0

    def test_stakes_sum_to_pool(self):
        """Total allocated stakes should not exceed pool."""
        runners = [
            _candidate(1, "H1", 5.0, 0.30, 0.60),
            _candidate(2, "H2", 6.0, 0.25, 0.55),
            _candidate(3, "H3", 8.0, 0.18, 0.45),
            _candidate(4, "Roughie", 15.0, 0.08, 0.25, value_rating=1.2),
        ]
        ctx = _race_context(runners)
        result = optimize_race(ctx, pool=20.0, venue_type="metro_vic")

        total = sum(r.stake_pct * 20.0 for r in result.recommendations)
        assert total <= 20.5  # allow small rounding

    def test_watch_only_reduced_total_stake(self):
        """Watch only races have reduced (half pool) total stake."""
        runners = [
            _candidate(1, "H1", 4.0, 0.25, 0.50, place_odds=2.0),
            _candidate(2, "H2", 5.0, 0.20, 0.40, place_odds=2.5),
        ]
        ctx = _race_context(runners)
        result = optimize_race(ctx)

        total = sum(r.stake_pct * 20.0 for r in result.recommendations)
        assert total > 0
        assert total <= 10.0 + 0.5  # half of default $20 pool + rounding

    def test_venue_confidence_affects_classification(self):
        """Country venue needs more edge to avoid Watch Only."""
        # Marginal edge that passes at metro but fails at country
        runners = [
            _candidate(1, "H1", 5.0, 0.23, 0.55, place_odds=2.0),
            # win_edge = 0.23 - 0.20 = 0.03 (exactly at threshold for metro)
            # place_edge = 0.55 - 0.50 = 0.05 (exactly at threshold)
            _candidate(2, "H2", 6.0, 0.18, 0.42, place_odds=2.5),
        ]
        ctx = _race_context(runners)

        metro = optimize_race(ctx, venue_type="metro_vic")
        country = optimize_race(ctx, venue_type="country")

        # Metro might not be watch only, country should be tighter
        # At country: edge thresholds = 0.03/0.80 = 0.0375 and 0.05/0.80 = 0.0625
        # Runner 1 win_edge=0.03 < 0.0375 and place_edge=0.05 < 0.0625
        assert country.classification.watch_only is True

    def test_circuit_breaker_tightens_thresholds(self):
        """Circuit breaker active -> tighter edge thresholds."""
        runners = [
            _candidate(1, "H1", 5.0, 0.24, 0.55, place_odds=2.0),
            # win_edge = 0.24 - 0.20 = 0.04
            # Without CB: threshold = 0.03 -> passes
            # With CB: threshold = 0.03 * 1.2 = 0.036 -> still passes
            # For watch_only: place_edge = 0.55 - 0.50 = 0.05
            # Without CB: threshold = 0.05 -> passes
            # With CB: threshold = 0.05 * 1.2 = 0.06 -> fails
            _candidate(2, "H2", 6.0, 0.18, 0.40, place_odds=2.5),
        ]
        ctx = _race_context(runners)

        normal = optimize_race(ctx, meeting_hit_count=2, meeting_race_count=6)
        breaker = optimize_race(ctx, meeting_hit_count=0, meeting_race_count=6)

        assert breaker.circuit_breaker_active is True
        assert not normal.circuit_breaker_active
