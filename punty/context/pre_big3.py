"""Pre-calculate optimal Big 3 Multi combination.

Finds the 3-horse combination across different races that maximises
the product of win probabilities while ensuring the combined multi
odds still offer value (positive expected value).

The AI receives this as a RECOMMENDED default and can override
with justification.
"""

import logging
from dataclasses import dataclass
from itertools import combinations

logger = logging.getLogger(__name__)

MULTI_STAKE = 10.0
# Minimum win probability for a horse to be eligible for Big 3
MIN_WIN_PROB = 0.15
# Minimum multi EV ratio (expected_payout / stake) to recommend
MIN_MULTI_EV = 1.0
# Pool takeout estimate for multi bets
POOL_TAKEOUT = 0.85


@dataclass
class Big3Candidate:
    """A single horse candidate for the Big 3."""

    race_number: int
    saddlecloth: int
    horse_name: str
    win_prob: float
    win_odds: float
    tip_rank: int  # 1-4 from pre-selections
    reason: str  # one-line reason from pre-selections


@dataclass
class Big3Recommendation:
    """The recommended Big 3 Multi combination."""

    horses: list[Big3Candidate]
    multi_prob: float  # product of win probs
    multi_odds: float  # product of decimal odds
    expected_value: float  # multi_prob * multi_odds * POOL_TAKEOUT
    stake: float
    estimated_return: float  # EV * stake


def calculate_pre_big3(race_contexts: list[dict]) -> Big3Recommendation | None:
    """Find the optimal Big 3 combination from all race pre-selections.

    Args:
        race_contexts: List of race context dicts, each with 'pre_selections'
                       and 'probabilities' data.

    Returns:
        Big3Recommendation or None if no valid combination found.
    """
    # Collect top candidates from each race
    candidates_by_race: dict[int, list[Big3Candidate]] = {}

    for rc in race_contexts:
        rn = rc.get("race_number")
        if not rn:
            continue

        pre_sel = rc.get("pre_selections")
        probs = rc.get("probabilities", {})
        ranked = probs.get("probability_ranked", [])

        if not ranked:
            continue

        # Build probability lookup
        prob_lookup: dict[int, float] = {}
        for entry in ranked:
            sc = entry.get("saddlecloth")
            wp = entry.get("win_prob", 0)
            if isinstance(wp, str):
                wp = float(wp.rstrip("%")) / 100
            if sc:
                prob_lookup[sc] = float(wp)

        # Get top picks from pre-selections (up to top 2 per race)
        picks = []
        if pre_sel and "picks" in pre_sel:
            for p in pre_sel["picks"][:2]:  # top 2 only
                sc = p.get("saddlecloth")
                if not sc:
                    continue
                wp = prob_lookup.get(sc, 0)
                if wp < MIN_WIN_PROB:
                    continue

                odds = p.get("win_odds") or p.get("odds", 0)
                if not odds or odds <= 1:
                    continue

                picks.append(Big3Candidate(
                    race_number=rn,
                    saddlecloth=sc,
                    horse_name=p.get("horse_name", ""),
                    win_prob=wp,
                    win_odds=float(odds),
                    tip_rank=p.get("rank", 1),
                    reason=p.get("reason", ""),
                ))

        # Fallback: use probability_ranked directly if no pre-selections
        if not picks:
            for entry in ranked[:2]:
                sc = entry.get("saddlecloth")
                wp = prob_lookup.get(sc, 0)
                if wp < MIN_WIN_PROB and not picks:
                    continue

                # Find odds from runners
                odds = 0
                for r in rc.get("runners", []):
                    r_sc = r.get("saddlecloth") or r.get("saddle_cloth")
                    if r_sc == sc:
                        odds = r.get("current_odds") or r.get("win_odds", 0)
                        horse_name = r.get("horse_name", entry.get("horse", ""))
                        break
                else:
                    horse_name = entry.get("horse", "")
                    odds = entry.get("odds", 0)

                if not odds or odds <= 1:
                    continue

                picks.append(Big3Candidate(
                    race_number=rn,
                    saddlecloth=sc,
                    horse_name=horse_name,
                    win_prob=wp,
                    win_odds=float(odds),
                    tip_rank=1,
                    reason="",
                ))

        if picks:
            candidates_by_race[rn] = picks

    if len(candidates_by_race) < 3:
        logger.debug(f"Only {len(candidates_by_race)} races with candidates, need 3")
        return None

    # Find optimal 3-race combination
    # For each combo of 3 races, try all candidate combinations
    best: Big3Recommendation | None = None
    best_ev = 0.0

    race_numbers = sorted(candidates_by_race.keys())

    for race_combo in combinations(race_numbers, 3):
        # Get candidate lists for these 3 races
        lists = [candidates_by_race[rn] for rn in race_combo]

        # Try all combinations of candidates
        for c1 in lists[0]:
            for c2 in lists[1]:
                for c3 in lists[2]:
                    horses = [c1, c2, c3]
                    multi_prob = c1.win_prob * c2.win_prob * c3.win_prob
                    multi_odds = c1.win_odds * c2.win_odds * c3.win_odds

                    # Expected value: probability * payout * pool_takeout
                    ev = multi_prob * multi_odds * POOL_TAKEOUT

                    if ev > best_ev:
                        best_ev = ev
                        best = Big3Recommendation(
                            horses=horses,
                            multi_prob=multi_prob,
                            multi_odds=multi_odds,
                            expected_value=ev,
                            stake=MULTI_STAKE,
                            estimated_return=ev * MULTI_STAKE,
                        )

    if not best:
        return None

    if best.expected_value < MIN_MULTI_EV:
        logger.info(
            f"Best Big 3 EV {best.expected_value:.2f} below minimum {MIN_MULTI_EV}"
        )
        # Still return it but log the warning — let the AI decide
        pass

    logger.info(
        f"Big 3 recommendation: "
        f"{' + '.join(h.horse_name for h in best.horses)} "
        f"multi_prob={best.multi_prob:.4f} "
        f"multi_odds={best.multi_odds:.1f} "
        f"EV={best.expected_value:.2f}"
    )

    return best


def format_pre_big3(rec: Big3Recommendation) -> str:
    """Format the Big 3 recommendation for injection into AI prompt context."""
    if not rec:
        return ""

    lines = ["\n**PRE-CALCULATED BIG 3 MULTI (recommended combination):**"]
    lines.append(
        f"This combination maximises the multi's expected value "
        f"by selecting the 3 horses across different races with the "
        f"best probability × odds product."
    )
    lines.append("")

    for i, h in enumerate(rec.horses, 1):
        ev_flag = ""
        if h.win_prob >= 0.30:
            ev_flag = " [STRONG]"
        elif h.win_prob >= 0.20:
            ev_flag = " [SOLID]"
        lines.append(
            f"  {i}) {h.horse_name} (Race {h.race_number}, No.{h.saddlecloth}) "
            f"— ${h.win_odds:.2f} "
            f"(win prob: {h.win_prob:.0%}){ev_flag}"
        )
        if h.reason:
            lines.append(f"     {h.reason}")

    lines.append("")
    lines.append(
        f"  Multi probability: {rec.multi_prob:.1%} "
        f"| Multi odds: ~${rec.multi_odds:.0f} "
        f"| EV ratio: {rec.expected_value:.2f}x"
    )

    if rec.expected_value >= 1.2:
        lines.append("  VALUE: Strong positive EV — recommend this multi.")
    elif rec.expected_value >= 1.0:
        lines.append("  VALUE: Fair value — proceed with this multi.")
    else:
        lines.append(
            f"  WARNING: Below fair value (EV {rec.expected_value:.2f}x). "
            f"Consider skipping the multi or finding better candidates."
        )

    lines.append("")
    lines.append(
        "  Use these exact horses unless you have strong reason to override. "
        "The combination is mathematically optimised for maximum EV."
    )

    return "\n".join(lines)
