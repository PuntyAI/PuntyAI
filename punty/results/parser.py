"""Enhanced early mail parser — extracts all pick types from generated content."""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- Big 3 section ---
_BIG3_SECTION = re.compile(
    r"\*{1,2}(?:PUNTY['\u2019]?S?\s+)?BIG\s*3.*?\*{1,2}\s*\n(.*?)(?=\n###|\n\*{1,2}RACE|\n\*{1,2}Race\s+\d|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_BIG3_HORSE = re.compile(
    # Matches both formats:
    #   Old: 1) *HORSE* (Race 2, No.8) — $2.80
    #   New: *1 - HORSE* (Race 2, No.8) — $2.80
    r"(?:(?P<rank_old>\d)\)\s*\*(?P<name_old>[A-Z][A-Z\s'\u2019\-]+?)\*|\*(?P<rank_new>\d)\s*[-–—]\s*(?P<name_new>[A-Z][A-Z\s'\u2019\-]+?)\*)\s*\(Race\s*(?P<race>\d+),\s*No\.(?P<saddle>\d+)\)\s*.*?\$(?P<odds>\d+\.?\d*)",
    re.IGNORECASE,
)
_BIG3_MULTI = re.compile(
    r"Multi.*?(\d+)U\s*[×x]\s*~?\$?(\d+\.?\d*)\s*=\s*~?(\d+\.?\d*)U",
    re.IGNORECASE,
)

# --- Race selections ---
_RACE_HEADER = re.compile(
    r"\*Race\s+(\d+)\s*[–\-—]", re.IGNORECASE,
)
# Matches both formats:
#   Old: 1. *HORSE* (No.1) — $3.50 / $1.45
#   New: *1. HORSE* (No.1) — $3.50 / $1.45
_SELECTION = re.compile(
    r"(?:(\d)\.\s*\*([A-Z][A-Z\s'\u2019\-]+?)\*|\*(\d)\.\s*([A-Z][A-Z\s'\u2019\-]+?)\*)\s*\(No\.(\d+)\)\s*.*?\$(\d+\.?\d*)(?:\s*/\s*\$(\d+\.?\d*))?",
    re.IGNORECASE,
)
# Matches both formats:
#   Old: Roughie: *HORSE* (No.1) — $3.50 / $1.45
#   New: *Roughie: HORSE* (No.1) — $3.50 / $1.45
_ROUGHIE = re.compile(
    r"(?:Roughie:\s*\*([A-Z][A-Z\s'\u2019\-]+?)\*|\*Roughie:\s*([A-Z][A-Z\s'\u2019\-]+?)\*)\s*\(No\.(\d+)\)\s*.*?\$(\d+\.?\d*)(?:\s*/\s*\$(\d+\.?\d*))?",
    re.IGNORECASE,
)

# --- Bet line (for selections) ---
_BET_LINE = re.compile(
    r"Bet:\s*\$(\d+\.?\d*)\s*(Win\s*\(Saver\)|Win|Saver\s*Win|Place|Each\s*Way|E/W)",
    re.IGNORECASE,
)
# "Bet: Exotics only" — no stake
_BET_EXOTICS_ONLY = re.compile(r"Bet:\s*Exotics\s*only", re.IGNORECASE)

# --- Confidence & Probability ---
_CONFIDENCE = re.compile(
    r"Confidence:\s*(HIGH|MED|MEDIUM|LOW)",
    re.IGNORECASE,
)
_PROBABILITY = re.compile(
    r"Probability:\s*(\d+\.?\d*)%",
    re.IGNORECASE,
)
_VALUE_RATING = re.compile(
    r"Value:\s*(\d+\.?\d*)x",
    re.IGNORECASE,
)

# --- Exotics ---
# Matches formats like:
#   Exacta: 1, 2, 4 — $20
#   Exacta: 1 to win, 2, 4 for second — $20
#   Trifecta: 1/2,4/3,5,6 — $20
_EXOTIC = re.compile(
    r"\*?Degenerate\s+Exotic.*?\*?\s*\n\s*"
    r"((?:Trifecta|Exacta|Quinella|First\s*(?:Four|4))(?:\s*\(?\s*(?:Standout|Box(?:ed)?)\s*\)?)?|"
    r"(?:Box(?:ed)?\s+)?(?:Trifecta|Exacta|Quinella|First\s*(?:Four|4))|"
    r"Standout\s+(?:Exacta|Quinella|Trifecta|First\s*(?:Four|4))|"
    r"Trifecta\s+Standout|First4)"
    r":\s*(.+?)\s*"  # Capture everything up to the stake
    r"(?:[–\-—]\s*(?:\$(\d+\.?\d*)|(\d+\.?\d*)U)|"    # — $20 or — 2U
    r"\(\$(\d+\.?\d*)\))",                               # ($20)
    re.IGNORECASE,
)
_EXOTIC_RETURN = re.compile(
    r"Est\.\s*return:\s*(\d+\.?\d*)%",
    re.IGNORECASE,
)

# --- Sequences ---
_SEQ_HEADER = re.compile(
    r"(EARLY\s+QUADDIE|MAIN\s+QUADDIE|QUADDIE|BIG\s*6)\s*\((?:Races?\s*)?R?(\d+)[–\-—](?:R)?(\d+)\)",
    re.IGNORECASE,
)
_SEQ_VARIANT = re.compile(
    r"(Skinny|Balanced|Wide)\s*(?:\(\$[\d.]+\))?:\s*(.+)",
    re.IGNORECASE,
)
# --- Punty's Pick (highlighted best-bet per race) ---
_PUNTYS_PICK = re.compile(
    r"\*?Punty'?s\s+Pick:?\*?\s*(.+)",
    re.IGNORECASE,
)
_PUNTYS_PICK_HORSE = re.compile(
    r"(?:^|[+])\s*(?:\*?([A-Z][A-Z\s'\u2019\-]+?)\*?\s*)?\(No\.(\d+)\)",
    re.IGNORECASE,
)
# Extract bet type + odds per horse from Punty's Pick line:
#   "HORSE (No.5) $1.37 Place" or "HORSE (No.5) $2.10 Win"
_PUNTYS_PICK_BET = re.compile(
    r"\(No\.(\d+)\)\s*\$(\d+\.?\d*)\s*(Win\s*\(Saver\)|Win|Saver\s*Win|Place|Each\s*Way|E/W)",
    re.IGNORECASE,
)
# Exotic Punty's Pick format:
#   *Punty's Pick:* Trifecta Box [1, 5, 8, 12] — $20 (Value: 1.8x)
_PUNTYS_PICK_EXOTIC = re.compile(
    r"(Trifecta\s+Box|Trifecta\s+Standout|Exacta|Quinella|First4\s+Box|First4)\s*"
    r"\[([^\]]+)\]\s*"
    r"[–\-—]\s*\$(\d+\.?\d*)"
    r"(?:\s*\(Value:\s*(\d+\.?\d*)x\))?",
    re.IGNORECASE,
)

_SEQ_COSTING = re.compile(
    # Handles two formats:
    # "(729 combos × $0.14 = $100) — est. return: 14%"
    # "(1×1×1×1 = 1 combo × $1.00 = $1) — est. return: 100%"
    r"\((?:[\d×x]+\s*=\s*)?(\d+)\s*combos?\s*[×x]\s*\$(\d+\.?\d*)\s*=\s*\$(\d+\.?\d*)\)\s*(?:[–\-—]\s*est\.\s*return:\s*(\d+\.?\d*)%)?",
    re.IGNORECASE,
)


def _normalize_bet_type(raw: str) -> str:
    """Normalize bet type string to snake_case."""
    t = raw.strip().lower()
    if t in ("e/w", "each way"):
        return "each_way"
    if t in ("win (saver)", "saver win"):
        return "saver_win"
    return t.replace(" ", "_")


def _normalize_exotic_type(raw: str) -> str:
    """Normalize exotic type to canonical name for consistent tracking.

    Canonical names: Exacta, Quinella, Trifecta Box, First4 Box
    """
    t = raw.strip().lower()
    # Remove parentheses and extra whitespace
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()

    if "first" in t and ("4" in t or "four" in t):
        if "box" in t:
            return "First4 Box"
        return "First4"
    if "trifecta" in t:
        if "standout" in t:
            return "Trifecta Standout"
        return "Trifecta Box"
    if "exacta" in t:
        if "standout" in t:
            return "Exacta"  # Standout exacta is still a straight exacta
        return "Exacta"
    if "quinella" in t or "quin" in t:
        return "Quinella"
    return raw.strip()


def parse_early_mail(raw_content: str, content_id: str, meeting_id: str) -> list[dict]:
    """Parse early mail content into a list of pick dicts ready for DB insertion."""
    if not raw_content:
        return []

    picks: list[dict] = []
    _id_counter = 0

    def _next_id() -> str:
        nonlocal _id_counter
        _id_counter += 1
        return f"pk-{content_id[:8]}-{_id_counter:03d}"

    # --- Big 3 ---
    picks.extend(_parse_big3(raw_content, content_id, meeting_id, _next_id))

    # --- Race-by-race selections + exotics ---
    picks.extend(_parse_race_sections(raw_content, content_id, meeting_id, _next_id))

    # --- Sequences ---
    picks.extend(_parse_sequences(raw_content, content_id, meeting_id, _next_id))

    return picks


def _parse_big3(raw_content: str, content_id: str, meeting_id: str, next_id) -> list[dict]:
    picks = []
    section_m = _BIG3_SECTION.search(raw_content)
    if not section_m:
        return picks

    section = section_m.group(1)

    for m in _BIG3_HORSE.finditer(section):
        rank = m.group("rank_old") or m.group("rank_new")
        name = m.group("name_old") or m.group("name_new")
        picks.append({
            "id": next_id(),
            "content_id": content_id,
            "meeting_id": meeting_id,
            "race_number": int(m.group("race")),
            "horse_name": name.strip(),
            "saddlecloth": int(m.group("saddle")),
            "tip_rank": int(rank),
            "odds_at_tip": float(m.group("odds")),
            "place_odds_at_tip": None,
            "pick_type": "big3",
            "bet_type": None,
            "bet_stake": None,
            "exotic_type": None,
            "exotic_runners": None,
            "exotic_stake": None,
            "sequence_type": None,
            "sequence_variant": None,
            "sequence_legs": None,
            "sequence_start_race": None,
            "multi_odds": None,
            "estimated_return_pct": None,
        })

    multi_m = _BIG3_MULTI.search(section)
    if multi_m:
        picks.append({
            "id": next_id(),
            "content_id": content_id,
            "meeting_id": meeting_id,
            "race_number": None,
            "horse_name": None,
            "saddlecloth": None,
            "tip_rank": None,
            "odds_at_tip": None,
            "place_odds_at_tip": None,
            "pick_type": "big3_multi",
            "bet_type": None,
            "bet_stake": None,
            "exotic_type": None,
            "exotic_runners": None,
            "exotic_stake": float(multi_m.group(1)),
            "sequence_type": None,
            "sequence_variant": None,
            "sequence_legs": None,
            "sequence_start_race": None,
            "multi_odds": float(multi_m.group(2)),
            "estimated_return_pct": None,
        })

    return picks


def _parse_race_sections(raw_content: str, content_id: str, meeting_id: str, next_id) -> list[dict]:
    picks = []

    # Split content into race sections
    race_splits = _RACE_HEADER.split(raw_content)
    # race_splits alternates: [before_first_race, race_num_1, section_1, race_num_2, section_2, ...]
    if len(race_splits) < 3:
        return picks

    for i in range(1, len(race_splits), 2):
        race_num = int(race_splits[i])
        section = race_splits[i + 1] if i + 1 < len(race_splits) else ""

        # Selections (rank 1-3)
        for m in _SELECTION.finditer(section):
            # Look for bet line and metadata in the text after this match
            after_text = section[m.end():m.end() + 300]
            bet_m = _BET_LINE.search(after_text)
            if bet_m:
                bet_stake = float(bet_m.group(1))
                bet_type = _normalize_bet_type(bet_m.group(2))
            elif _BET_EXOTICS_ONLY.search(after_text):
                bet_stake = 0.0
                bet_type = "exotics_only"
            else:
                bet_stake = None
                bet_type = None
            # Extract confidence and probability
            confidence = _extract_confidence(after_text)
            probability = _extract_probability(after_text)
            value_rating = _extract_value_rating(after_text)
            # Handle both old (groups 1-5) and new (groups 3-7) format
            rank = m.group(1) or m.group(3)
            name = m.group(2) or m.group(4)
            saddle = m.group(5)
            win_odds = float(m.group(6))
            place_odds = float(m.group(7)) if m.group(7) else None
            picks.append({
                "id": next_id(),
                "content_id": content_id,
                "meeting_id": meeting_id,
                "race_number": race_num,
                "horse_name": name.strip(),
                "saddlecloth": int(saddle),
                "tip_rank": int(rank),
                "odds_at_tip": win_odds,
                "place_odds_at_tip": place_odds,
                "pick_type": "selection",
                "bet_type": bet_type,
                "bet_stake": bet_stake,
                "exotic_type": None,
                "exotic_runners": None,
                "exotic_stake": None,
                "sequence_type": None,
                "sequence_variant": None,
                "sequence_legs": None,
                "sequence_start_race": None,
                "multi_odds": None,
                "estimated_return_pct": None,
                "confidence": confidence,
                "win_probability": probability,
                "value_rating": value_rating,
            })

        # Roughie (rank 4)
        roughie_m = _ROUGHIE.search(section)
        if roughie_m:
            after_text = section[roughie_m.end():roughie_m.end() + 300]
            bet_m = _BET_LINE.search(after_text)
            if bet_m:
                bet_stake = float(bet_m.group(1))
                bet_type = _normalize_bet_type(bet_m.group(2))
            elif _BET_EXOTICS_ONLY.search(after_text):
                bet_stake = 0.0
                bet_type = "exotics_only"
            else:
                bet_stake = None
                bet_type = None
            # Extract confidence and probability
            confidence = _extract_confidence(after_text)
            probability = _extract_probability(after_text)
            value_rating = _extract_value_rating(after_text)
            # Handle both old (groups 1,2,3,4,5) and new (groups 2,3,4,5) format
            roughie_name = roughie_m.group(1) or roughie_m.group(2)
            roughie_saddle = roughie_m.group(3)
            win_odds = float(roughie_m.group(4))
            place_odds = float(roughie_m.group(5)) if roughie_m.group(5) else None
            picks.append({
                "id": next_id(),
                "content_id": content_id,
                "meeting_id": meeting_id,
                "race_number": race_num,
                "horse_name": roughie_name.strip(),
                "saddlecloth": int(roughie_saddle),
                "tip_rank": 4,
                "odds_at_tip": win_odds,
                "place_odds_at_tip": place_odds,
                "pick_type": "selection",
                "bet_type": bet_type,
                "bet_stake": bet_stake,
                "exotic_type": None,
                "exotic_runners": None,
                "exotic_stake": None,
                "sequence_type": None,
                "sequence_variant": None,
                "sequence_legs": None,
                "sequence_start_race": None,
                "multi_odds": None,
                "estimated_return_pct": None,
                "confidence": confidence,
                "win_probability": probability,
                "value_rating": value_rating,
            })

        # Punty's Pick — mark the highlighted best-bet selection(s) or create exotic pick
        puntys_pick_m = _PUNTYS_PICK.search(section)
        if puntys_pick_m:
            pick_line = puntys_pick_m.group(1)

            # Check for exotic Punty's Pick format first
            exotic_pp_m = _PUNTYS_PICK_EXOTIC.search(pick_line)
            if exotic_pp_m:
                exotic_type = _normalize_exotic_type(exotic_pp_m.group(1))
                runners = [int(x) for x in re.findall(r'\d+', exotic_pp_m.group(2))]
                stake = float(exotic_pp_m.group(3))
                value = float(exotic_pp_m.group(4)) if exotic_pp_m.group(4) else None
                picks.append({
                    "id": next_id(),
                    "content_id": content_id,
                    "meeting_id": meeting_id,
                    "race_number": race_num,
                    "horse_name": None,
                    "saddlecloth": None,
                    "tip_rank": None,
                    "odds_at_tip": None,
                    "place_odds_at_tip": None,
                    "pick_type": "exotic",
                    "bet_type": None,
                    "bet_stake": None,
                    "exotic_type": exotic_type,
                    "exotic_runners": json.dumps(runners),
                    "exotic_stake": stake,
                    "is_puntys_pick": True,
                    "value_rating": value,
                    "sequence_type": None,
                    "sequence_variant": None,
                    "sequence_legs": None,
                    "sequence_start_race": None,
                    "multi_odds": None,
                    "estimated_return_pct": None,
                })
            else:
                # Selection Punty's Pick — mark matching selections
                # Also extract bet type per horse (may differ from main selection)
                puntys_saddlecloths = set()
                pp_bet_overrides = {}  # {saddlecloth: (odds, bet_type)}
                for hm in _PUNTYS_PICK_HORSE.finditer(pick_line):
                    puntys_saddlecloths.add(int(hm.group(2)))
                for bm in _PUNTYS_PICK_BET.finditer(pick_line):
                    sc = int(bm.group(1))
                    odds = float(bm.group(2))
                    bt = _normalize_bet_type(bm.group(3))
                    pp_bet_overrides[sc] = (odds, bt)
                for p in picks:
                    if (p.get("race_number") == race_num
                            and p.get("pick_type") == "selection"
                            and p.get("saddlecloth") in puntys_saddlecloths):
                        p["is_puntys_pick"] = True
                        # Override bet type if Punty's Pick recommends differently
                        override = pp_bet_overrides.get(p["saddlecloth"])
                        if override:
                            pp_odds, pp_bet_type = override
                            if pp_bet_type != p.get("bet_type"):
                                p["pp_bet_type"] = pp_bet_type
                                p["pp_odds"] = pp_odds
                                logger.info(
                                    f"Punty's Pick R{race_num} #{p['saddlecloth']}: "
                                    f"overrides {p.get('bet_type')} → {pp_bet_type} at ${pp_odds}"
                                )

        # Exotic
        exotic_m = _EXOTIC.search(section)
        if exotic_m:
            exotic_type = _normalize_exotic_type(exotic_m.group(1))
            runners_str = exotic_m.group(2).strip()
            # Parse runners — handles various formats:
            #   "1, 2, 4" (simple box)
            #   "1/2,4/3,5,6" (positional with slashes - standout/flexi)
            #   "1 / 2, 4" or "1 / 2, 4 / 3, 5" (with spaces around slashes)
            # Store as legs array for display and combo calculation
            if "/" in runners_str:
                # Split by "/" to get legs, then extract numbers from each leg
                legs = []
                for leg in runners_str.split("/"):
                    leg_runners = [int(x) for x in re.findall(r'\d+', leg)]
                    if leg_runners:
                        legs.append(leg_runners)
                runners = legs  # Store as [[1], [2, 4], [3, 5, 6]]
            else:
                # Simple list - all numbers are boxed together
                runners = [int(x) for x in re.findall(r'\d+', runners_str)]
            # $20 format: group 3 (— $X), group 4 (— XU), group 5 (($X))
            stake_str = exotic_m.group(3) or exotic_m.group(4) or exotic_m.group(5)
            stake = float(stake_str) if stake_str else 20.0
            # Est. return %
            est_return_m = _EXOTIC_RETURN.search(section)
            est_return_pct = float(est_return_m.group(1)) if est_return_m else None
            picks.append({
                "id": next_id(),
                "content_id": content_id,
                "meeting_id": meeting_id,
                "race_number": race_num,
                "horse_name": None,
                "saddlecloth": None,
                "tip_rank": None,
                "odds_at_tip": None,
                "place_odds_at_tip": None,
                "pick_type": "exotic",
                "bet_type": None,
                "bet_stake": None,
                "exotic_type": exotic_type,
                "exotic_runners": json.dumps(runners),
                "exotic_stake": stake,
                "estimated_return_pct": est_return_pct,
                "sequence_type": None,
                "sequence_variant": None,
                "sequence_legs": None,
                "sequence_start_race": None,
                "multi_odds": None,
            })

    return picks


def _parse_sequences(raw_content: str, content_id: str, meeting_id: str, next_id) -> list[dict]:
    picks = []

    # Find the sequence lanes section
    seq_section_m = re.search(
        r"\*SEQUENCE\s+LANES\*\s*\n(.*?)(?=\n###|\n\*NUGGETS|\Z)",
        raw_content, re.DOTALL | re.IGNORECASE,
    )
    if not seq_section_m:
        # Try without the heading wrapper
        seq_section_m = re.search(
            r"((?:EARLY\s+QUADDIE|MAIN\s+QUADDIE|QUADDIE|BIG\s*6)\s*\((?:Races?\s*)?R?\d+.*?)(?=\n###|\n\*NUGGETS|\Z)",
            raw_content, re.DOTALL | re.IGNORECASE,
        )
    if not seq_section_m:
        return picks

    seq_text = seq_section_m.group(1) if seq_section_m else ""

    # Find each sequence block
    headers = list(_SEQ_HEADER.finditer(seq_text))
    for idx, hdr in enumerate(headers):
        seq_name = hdr.group(1).strip().upper()
        start_race = int(hdr.group(2))
        end_race = int(hdr.group(3))

        # Normalize type
        if "EARLY" in seq_name:
            seq_type = "early_quaddie"
        elif "QUADDIE" in seq_name:
            seq_type = "quaddie"
        elif "BIG" in seq_name or "6" in seq_name:
            seq_type = "big6"
        else:
            seq_type = seq_name.lower().replace(" ", "_")

        # Get text between this header and the next
        block_start = hdr.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(seq_text)
        block = seq_text[block_start:block_end]

        for vm in _SEQ_VARIANT.finditer(block):
            variant = vm.group(1).strip().lower()
            legs_raw = vm.group(2).strip()

            # Parse costing info if present
            costing_m = _SEQ_COSTING.search(legs_raw)
            combo_count = int(costing_m.group(1)) if costing_m else None
            unit_price = float(costing_m.group(2)) if costing_m else None
            # Store total outlay (not unit price) for accurate settlement
            total_outlay = float(costing_m.group(3)) if costing_m else None
            est_return_pct = float(costing_m.group(4)) if costing_m and costing_m.group(4) else None

            # Strip costing suffix from legs before parsing
            if costing_m:
                legs_raw = legs_raw[:costing_m.start()].strip()

            legs = []
            for leg in legs_raw.split("/"):
                saddlecloths = [int(x.strip()) for x in leg.split(",") if x.strip().isdigit()]
                if saddlecloths:
                    legs.append(saddlecloths)

            picks.append({
                "id": next_id(),
                "content_id": content_id,
                "meeting_id": meeting_id,
                "race_number": None,
                "horse_name": None,
                "saddlecloth": None,
                "tip_rank": None,
                "odds_at_tip": None,
                "place_odds_at_tip": None,
                "pick_type": "sequence",
                "bet_type": None,
                "bet_stake": None,
                "exotic_type": None,
                "exotic_runners": None,
                "exotic_stake": total_outlay,  # Store total outlay, not unit price
                "estimated_return_pct": est_return_pct,
                "sequence_type": seq_type,
                "sequence_variant": variant,
                "sequence_legs": json.dumps(legs),
                "sequence_start_race": start_race,
                "multi_odds": None,
            })

    return picks


# ──────────────────────────────────────────────
# Confidence / Probability extraction helpers
# ──────────────────────────────────────────────

def _extract_confidence(text: str) -> Optional[str]:
    """Extract confidence tag (HIGH/MED/LOW) from text after a selection."""
    m = _CONFIDENCE.search(text)
    if m:
        tag = m.group(1).upper()
        return "MED" if tag == "MEDIUM" else tag
    return None


def _extract_probability(text: str) -> Optional[float]:
    """Extract probability percentage from text (e.g. 'Probability: 32%' → 0.32)."""
    m = _PROBABILITY.search(text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _extract_value_rating(text: str) -> Optional[float]:
    """Extract value rating from text (e.g. 'Value: 1.4x' → 1.4)."""
    m = _VALUE_RATING.search(text)
    if m:
        return float(m.group(1))
    return None
