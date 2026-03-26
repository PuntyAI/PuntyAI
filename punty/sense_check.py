"""4-model sense check — compare our prediction against PF, Market, and KASH.

Our model is 100% independent (no market/KASH/PF predictions in scoring).
This module cross-references our R1 pick against three external models
AFTER prediction, returning a consensus level that affects staking.

Models:
1. Us — LGBM + tuned probability engine (form, KRI, pace, connections)
2. PF — Punting Form AI score / assessed price
3. Market — current odds favourite
4. KASH — Betfair Data Scientists rated price

Consensus:
- 3-4 agree → BET (full Kelly)
- 2 agree including us → BET (0.85× Kelly)
- We disagree with all 3 → SKIP (our model is the outlier)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def sense_check_race(
    our_r1_saddlecloth: int,
    runners: list[Any],
) -> dict:
    """Compare our top pick against external models.

    Args:
        our_r1_saddlecloth: Saddlecloth of our predicted R1 pick
        runners: List of runner objects (ORM or dict) with odds, KASH, PF data

    Returns:
        {
            "consensus": "HIGH" | "MEDIUM" | "LOW",
            "action": "bet" | "reduce" | "skip",
            "kelly_mult": 1.0 | 0.85 | 0.0,
            "our_pick": saddlecloth,
            "market_pick": saddlecloth or None,
            "kash_pick": saddlecloth or None,
            "pf_pick": saddlecloth or None,
            "agreements": int,
            "detail": str,
        }
    """
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    result = {
        "consensus": "LOW",
        "action": "skip",
        "kelly_mult": 0.0,
        "our_pick": our_r1_saddlecloth,
        "market_pick": None,
        "kash_pick": None,
        "pf_pick": None,
        "agreements": 0,
        "detail": "",
    }

    if not runners:
        result["detail"] = "No runners"
        return result

    active = [r for r in runners if not _get(r, "scratched")]
    if not active:
        result["detail"] = "All scratched"
        return result

    # ── Market favourite: lowest current_odds ──
    market_runners = [(r, _get(r, "current_odds") or 999) for r in active
                      if (_get(r, "current_odds") or 0) > 1.0]
    if market_runners:
        market_fav = min(market_runners, key=lambda x: x[1])
        result["market_pick"] = _get(market_fav[0], "saddlecloth")

    # ── KASH favourite: lowest kash_rated_price ──
    kash_runners = [(r, _get(r, "kash_rated_price") or 999) for r in active
                    if (_get(r, "kash_rated_price") or 0) > 0]
    if kash_runners:
        kash_fav = min(kash_runners, key=lambda x: x[1])
        result["kash_pick"] = _get(kash_fav[0], "saddlecloth")

    # ── PF favourite: highest pf_ai_score (or lowest pf_ai_price) ──
    pf_runners = [(r, _get(r, "pf_ai_score") or 0) for r in active
                  if (_get(r, "pf_ai_score") or 0) > 0]
    if not pf_runners:
        # Fallback: use pf_assessed_price (lower = better)
        pf_runners = [(r, -(_get(r, "pf_assessed_price") or 999)) for r in active
                      if (_get(r, "pf_assessed_price") or 0) > 0]
    if pf_runners:
        pf_fav = max(pf_runners, key=lambda x: x[1])
        result["pf_pick"] = _get(pf_fav[0], "saddlecloth")

    # ── Count agreements with our pick ──
    our_sc = our_r1_saddlecloth
    others = [result["market_pick"], result["kash_pick"], result["pf_pick"]]
    others_valid = [o for o in others if o is not None]
    agree_with_us = sum(1 for o in others_valid if o == our_sc)

    result["agreements"] = agree_with_us

    if agree_with_us >= 2:
        result["consensus"] = "HIGH"
        result["action"] = "bet"
        result["kelly_mult"] = 1.0
        result["detail"] = f"{agree_with_us + 1}/4 models agree"
    elif agree_with_us == 1:
        result["consensus"] = "MEDIUM"
        result["action"] = "reduce"
        result["kelly_mult"] = 0.85
        # Which model agrees?
        agreeing = []
        if result["market_pick"] == our_sc:
            agreeing.append("Market")
        if result["kash_pick"] == our_sc:
            agreeing.append("KASH")
        if result["pf_pick"] == our_sc:
            agreeing.append("PF")
        result["detail"] = f"2/4 agree (us + {'+'.join(agreeing)})"
    else:
        result["consensus"] = "LOW"
        result["action"] = "skip"
        result["kelly_mult"] = 0.0
        result["detail"] = (
            f"We're the outlier — Market={result['market_pick']}, "
            f"KASH={result['kash_pick']}, PF={result['pf_pick']}"
        )
        logger.warning(
            f"Sense check SKIP: our pick SC{our_sc} disagrees with all external models. "
            f"{result['detail']}"
        )

    return result
