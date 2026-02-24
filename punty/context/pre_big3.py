"""Pre-calculate optimal Big 3 Multi combination.

Selects the 3 horses across different races with the highest win
probabilities to maximise multi hit rate (#23). Previously used
EV-optimized (prob × odds) which chased payouts over collection.

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

    # Find optimal 3-race combination by maximising multi probability (#23)
    # Pick the single best candidate per race (highest win_prob), then
    # choose the 3 races whose best candidates have the highest combined probability.
    best_per_race: dict[int, Big3Candidate] = {}
    for rn, cands in candidates_by_race.items():
        best_per_race[rn] = max(cands, key=lambda c: c.win_prob)

    # Sort races by their best candidate's win_prob descending
    sorted_races = sorted(best_per_race.keys(), key=lambda rn: best_per_race[rn].win_prob, reverse=True)

    if len(sorted_races) < 3:
        return None

    # Take top 3 races by probability
    top_horses = [best_per_race[rn] for rn in sorted_races[:3]]
    multi_prob = top_horses[0].win_prob * top_horses[1].win_prob * top_horses[2].win_prob
    multi_odds = top_horses[0].win_odds * top_horses[1].win_odds * top_horses[2].win_odds
    ev = multi_prob * multi_odds * POOL_TAKEOUT

    best = Big3Recommendation(
        horses=top_horses,
        multi_prob=multi_prob,
        multi_odds=multi_odds,
        expected_value=ev,
        stake=MULTI_STAKE,
        estimated_return=ev * MULTI_STAKE,
    )

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
        f"This combination selects the 3 most probable winners "
        f"across different races to maximise multi hit rate."
    )
    lines.append("")

    for i, h in enumerate(rec.horses, 1):
        prob_flag = ""
        if h.win_prob >= 0.30:
            prob_flag = " [STRONG]"
        elif h.win_prob >= 0.20:
            prob_flag = " [SOLID]"
        lines.append(
            f"  {i}) {h.horse_name} (Race {h.race_number}, No.{h.saddlecloth}) "
            f"— ${h.win_odds:.2f} "
            f"(win prob: {h.win_prob:.0%}){prob_flag}"
        )
        if h.reason:
            lines.append(f"     {h.reason}")

    lines.append("")
    lines.append(
        f"  Multi probability: {rec.multi_prob:.1%} "
        f"| Multi odds: ~${rec.multi_odds:.0f} "
        f"| EV ratio: {rec.expected_value:.2f}x"
    )

    lines.append("")
    lines.append(
        "  Use these exact horses — they are the 3 most likely winners on the card. "
        "Do NOT substitute lower-probability horses for higher odds."
    )

    return "\n".join(lines)
