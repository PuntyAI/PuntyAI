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
    # Saddlecloth accepts: No.X, No X, #X
    r"(?:(?P<rank_old>\d)\)\s*\*(?P<name_old>[A-Z][A-Z\s'\u2019\-]+?)\*|\*(?P<rank_new>\d)\s*[-–—]\s*(?P<name_new>[A-Z][A-Z\s'\u2019\-]+?)\*)\s*\(Race\s*(?P<race>\d+),\s*(?:No\.?\s*|#)(?P<saddle>\d+)\)\s*.*?\$(?P<odds>\d+\.?\d*)",
    re.IGNORECASE,
)
_BIG3_MULTI = re.compile(
    r"Multi.*?\$?(\d+)(?:U)?\s*[×x]\s*~?\$?(\d+\.?\d*)\s*=\s*~?\$?(\d+\.?\d*)\s*(?:U|collect)?",
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
    r"(?:(\d)\.\s*\*([A-Z][A-Z\s'\u2019\-]+?)\*|\*(\d)\.\s*([A-Z][A-Z\s'\u2019\-]+?)\*)\s*\((?:No\.?\s*|#)(\d+)\)\s*.*?\$(\d+\.?\d*)(?:\s*/\s*\$(\d+\.?\d*))?",
    re.IGNORECASE,
)
# Matches both formats:
#   Old: Roughie: *HORSE* (No.1) — $3.50 / $1.45
#   New: *Roughie: HORSE* (No.1) — $3.50 / $1.45
_ROUGHIE = re.compile(
    r"(?:Roughie:\s*\*([A-Z][A-Z\s'\u2019\-]+?)\*|\*Roughie:\s*([A-Z][A-Z\s'\u2019\-]+?)\*)\s*\((?:No\.?\s*|#)(\d+)\)\s*.*?\$(\d+\.?\d*)(?:\s*/\s*\$(\d+\.?\d*))?",
    re.IGNORECASE,
)

# --- Bet line (for selections) ---
_BET_LINE = re.compile(
    r"Bet:\s*\$(\d+\.?\d*)\s*(Win\s*\(Saver\)|Win|Saver\s*Win|Place|Each\s*Way|E/W)",
    re.IGNORECASE,
)
# "Bet: Exotics only" — no stake
_BET_EXOTICS_ONLY = re.compile(r"Bet:\s*Exotics\s*only", re.IGNORECASE)
# "Bet: No Bet (Tracked)" or "Bet: No Bet" or "Bet: No Bet — reason"
_BET_TRACKED = re.compile(
    r"Bet:\s*No\s*Bet(?:\s*\(Tracked\))?(?:\s*[—–\-]\s*(.+))?",
    re.IGNORECASE,
)

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
# Primary exotic pattern: with "Degenerate Exotic" header
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
# Fallback exotic pattern: no header required, matches exotic type at line start
_EXOTIC_FALLBACK = re.compile(
    r"^\s*"
    r"((?:Trifecta|Exacta|Quinella|First\s*(?:Four|4))(?:\s*\(?\s*(?:Standout|Box(?:ed)?)\s*\)?)?|"
    r"(?:Box(?:ed)?\s+)?(?:Trifecta|Exacta|Quinella|First\s*(?:Four|4))|"
    r"Standout\s+(?:Exacta|Quinella|Trifecta|First\s*(?:Four|4))|"
    r"Trifecta\s+Standout|First4)"
    r":\s*(.+?)\s*"
    r"(?:[–\-—]\s*(?:\$(\d+\.?\d*)|(\d+\.?\d*)U)|"
    r"\(\$(\d+\.?\d*)\))",
    re.IGNORECASE | re.MULTILINE,
)
_EXOTIC_RETURN = re.compile(
    r"Est\.\s*return:\s*(\d+\.?\d*)%",
    re.IGNORECASE,
)

# --- Sequences ---
_SEQ_HEADER = re.compile(
    r"(EARLY\s+QUADDIE|MAIN\s+QUADDIE|QUADDIE|BIG\s*6)\s*(?:\(main\)\s*)?(?:\((?:Races?\s*)?R?(\d+)[–\-—](?:R)?(\d+)\)|:?\s*Races?\s*R?(\d+)[–\-—]R?(\d+))",
    re.IGNORECASE,
)
_SEQ_VARIANT = re.compile(
    r"(Skinny|Balanced|Wide|Smart|Ticket\s*[ABC])\s*(?:\(\$[\d.]+\))?:\s*(.+)",
    re.IGNORECASE,
)
# --- Punty's Pick (highlighted best-bet per race) ---
_PUNTYS_PICK = re.compile(
    r"\*?Punty['\u2019]?s\s+Pick:?\*?\s*(.+)",
    re.IGNORECASE,
)
_PUNTYS_PICK_HORSE = re.compile(
    r"(?:^|[+])\s*(?:\*?([A-Z][A-Z\s'\u2019\-]+?)\*?\s*)?\((?:No\.?\s*|#)(\d+)\)",
    re.IGNORECASE,
)
# Extract bet type + odds per horse from Punty's Pick line:
#   "HORSE (No.5) $1.37 Place" or "HORSE (No.5) $2.10 Win"
_PUNTYS_PICK_BET = re.compile(
    r"\((?:No\.?\s*|#)(\d+)\)\s*\$(\d+\.?\d*)\s*(Win\s*\(Saver\)|Win|Saver\s*Win|Place|Each\s*Way|E/W)",
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
            return "Exacta Standout"
        return "Exacta"
    if "quinella" in t or "quin" in t:
        return "Quinella"
    return raw.strip()


def _try_parse_json_block(raw_content: str, content_id: str, meeting_id: str) -> list[dict] | None:
    """Try to extract picks from JSON block in AI output. Returns None if no JSON found."""
    json_match = re.search(r'```json\s*\n(.*?)\n```', raw_content, re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        logger.warning("JSON block found but failed to parse")
        return None

    picks: list[dict] = []
    _id_counter = 0

    def _next_id() -> str:
        nonlocal _id_counter
        _id_counter += 1
        return f"pk-{content_id[:8]}-{_id_counter:03d}"

    def _pick_base() -> dict:
        return {
            "content_id": content_id,
            "meeting_id": meeting_id,
            "race_number": None,
            "horse_name": None,
            "saddlecloth": None,
            "tip_rank": None,
            "odds_at_tip": None,
            "place_odds_at_tip": None,
            "pick_type": None,
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
        }

    # --- Big 3 ---
    for b in data.get("big3", []):
        p = _pick_base()
        p["id"] = _next_id()
        p["pick_type"] = "big3"
        p["race_number"] = b.get("race")
        p["horse_name"] = b.get("horse")
        p["saddlecloth"] = b.get("saddlecloth")
        p["tip_rank"] = b.get("rank")
        p["odds_at_tip"] = b.get("odds")
        picks.append(p)

    # --- Big 3 Multi ---
    b3m = data.get("big3_multi")
    if b3m and b3m.get("stake"):
        p = _pick_base()
        p["id"] = _next_id()
        p["pick_type"] = "big3_multi"
        p["exotic_stake"] = b3m.get("stake")
        p["multi_odds"] = b3m.get("multi_odds")
        picks.append(p)

    # --- Race selections + exotics ---
    for race_key, race_data in data.get("races", {}).items():
        race_num = int(race_key)

        for sel in race_data.get("selections", []):
            p = _pick_base()
            p["id"] = _next_id()
            p["pick_type"] = "selection"
            p["race_number"] = race_num
            p["horse_name"] = sel.get("horse")
            p["saddlecloth"] = sel.get("saddlecloth")
            p["tip_rank"] = sel.get("rank")
            p["odds_at_tip"] = sel.get("win_odds")
            p["place_odds_at_tip"] = sel.get("place_odds")
            p["bet_type"] = _normalize_bet_type(sel.get("bet_type", "place"))
            p["bet_stake"] = sel.get("stake")
            p["confidence"] = sel.get("confidence")
            raw_prob = sel.get("probability")
            # Normalise to 0-1 decimal (AI JSON sends percentage e.g. 95.0)
            p["win_probability"] = raw_prob / 100 if raw_prob and raw_prob > 1 else raw_prob
            p["value_rating"] = sel.get("value")
            picks.append(p)

        # Roughie
        roughie = race_data.get("roughie")
        if roughie and roughie.get("horse"):
            p = _pick_base()
            p["id"] = _next_id()
            p["pick_type"] = "selection"
            p["race_number"] = race_num
            p["horse_name"] = roughie.get("horse")
            p["saddlecloth"] = roughie.get("saddlecloth")
            p["tip_rank"] = 4
            p["odds_at_tip"] = roughie.get("win_odds")
            p["place_odds_at_tip"] = roughie.get("place_odds")
            bt = roughie.get("bet_type", "exotics_only")
            p["bet_type"] = _normalize_bet_type(bt) if bt != "exotics_only" else "exotics_only"
            p["bet_stake"] = roughie.get("stake", 0.0)
            picks.append(p)

        # Exotic
        exotic = race_data.get("exotic")
        if exotic and exotic.get("type"):
            p = _pick_base()
            p["id"] = _next_id()
            p["pick_type"] = "exotic"
            p["race_number"] = race_num
            p["exotic_type"] = _normalize_exotic_type(exotic["type"])
            runners = exotic.get("runners", [])
            p["exotic_runners"] = json.dumps(runners) if runners else None
            p["exotic_stake"] = exotic.get("stake")
            picks.append(p)

        # Punty's Pick — mark the selection or create exotic PP
        pp = race_data.get("puntys_pick")
        if pp:
            pp_sc = pp.get("saddlecloth")
            pp_exotic_type = pp.get("exotic_type")
            if pp_exotic_type:
                # Exotic Punty's Pick — create a separate exotic pick marked as PP
                ep = _pick_base()
                ep["id"] = _next_id()
                ep["pick_type"] = "exotic"
                ep["race_number"] = race_num
                ep["exotic_type"] = _normalize_exotic_type(pp_exotic_type)
                ep_runners = pp.get("runners", [])
                ep["exotic_runners"] = json.dumps(ep_runners) if ep_runners else None
                ep["exotic_stake"] = pp.get("stake")
                ep["is_puntys_pick"] = True
                ep["value_rating"] = pp.get("value")
                picks.append(ep)
            elif pp_sc:
                # Selection Punty's Pick — mark matching selection(s)
                pp_bt = pp.get("bet_type")
                pp_odds = pp.get("odds")
                for p in picks:
                    if (p.get("race_number") == race_num
                            and p.get("pick_type") == "selection"
                            and p.get("saddlecloth") == pp_sc):
                        p["is_puntys_pick"] = True
                        if pp_bt and _normalize_bet_type(pp_bt) != p.get("bet_type"):
                            p["pp_bet_type"] = _normalize_bet_type(pp_bt)
                            if pp_odds:
                                p["pp_odds"] = pp_odds

    # --- Sequences ---
    for seq in data.get("sequences", []):
        seq_type = seq.get("type", "").lower()
        start_race = seq.get("start_race")
        end_race = seq.get("end_race")
        for var in seq.get("variants", []):
            p = _pick_base()
            p["id"] = _next_id()
            p["pick_type"] = "sequence"
            p["sequence_type"] = "quaddie" if "quad" in seq_type else "big6"
            p["sequence_variant"] = var.get("name", "").lower()
            legs_raw = var.get("legs", "")
            # Parse "3,4,9/3,2,1/10,4,12,8" into [[3,4,9],[3,2,1],[10,4,12,8]]
            if isinstance(legs_raw, list):
                # Already a list of lists from AI JSON
                legs_parsed = legs_raw
            elif isinstance(legs_raw, str) and "/" in legs_raw:
                legs_parsed = []
                for leg in legs_raw.split("/"):
                    saddlecloths = [int(x.strip()) for x in leg.split(",") if x.strip().isdigit()]
                    if saddlecloths:
                        legs_parsed.append(saddlecloths)
            else:
                legs_parsed = None
            p["sequence_legs"] = json.dumps(legs_parsed) if legs_parsed else None
            p["sequence_start_race"] = start_race
            p["exotic_stake"] = var.get("total")
            p["estimated_return_pct"] = var.get("est_return")
            picks.append(p)

    if not picks:
        logger.warning("JSON block parsed but produced 0 picks")
        return None

    logger.info(f"Parsed {len(picks)} picks from JSON block")
    return picks


def parse_early_mail(raw_content: str, content_id: str, meeting_id: str) -> list[dict]:
    """Parse early mail content into a list of pick dicts ready for DB insertion."""
    if not raw_content:
        return []

    # Try JSON first (Phase 2 — deterministic parsing)
    json_picks = _try_parse_json_block(raw_content, content_id, meeting_id)
    if json_picks:
        return json_picks

    # Fall back to regex parsing
    logger.info("No JSON block found, using regex parser")

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

    # --- Completeness validation ---
    # Check which race numbers appear in Race headers vs parsed picks
    race_headers = set(int(m.group(1)) for m in _RACE_HEADER.finditer(raw_content))
    races_with_picks = set(p["race_number"] for p in picks if p.get("pick_type") == "selection")
    missing_races = race_headers - races_with_picks
    if missing_races:
        logger.warning(
            f"Parse completeness: {meeting_id} — race headers found for "
            f"R{sorted(missing_races)} but no selections parsed for those races"
        )
    races_without_exotics = set()
    races_with_exotics = set(p["race_number"] for p in picks if p.get("pick_type") == "exotic")
    if race_headers:
        races_without_exotics = race_headers - races_with_exotics
        if len(races_without_exotics) > len(race_headers) * 0.5:
            logger.warning(
                f"Parse completeness: {meeting_id} — {len(races_without_exotics)}/{len(race_headers)} "
                f"races have no exotic parsed"
            )

    return picks


def validate_parsed_picks(parsed_picks: list[dict], pre_selections: dict) -> list[str]:
    """Compare parsed picks against pre-selection recommendations. Return warnings.

    Cross-references parser output with what the pre-selection engine recommended,
    logging any silent failures where picks were expected but not parsed.
    """
    warnings = []
    expected_races = set(pre_selections.get("races", {}).keys())
    parsed_races = set(p.get("race_number") for p in parsed_picks if p.get("pick_type") == "selection")

    missing_races = expected_races - parsed_races
    for race_num in sorted(missing_races):
        race_data = pre_selections.get("races", {}).get(race_num, {})
        expected_count = len(race_data.get("picks", []))
        warnings.append(
            f"Race {race_num}: no picks parsed (expected {expected_count} selections)"
        )

    # Check Big3 Multi was parsed
    has_big3_multi = any(p.get("pick_type") == "big3_multi" for p in parsed_picks)
    if pre_selections.get("big3") and not has_big3_multi:
        warnings.append("Big3 Multi not parsed — check regex match")

    # Check exotics per race
    parsed_exotic_races = set(p.get("race_number") for p in parsed_picks if p.get("pick_type") == "exotic")
    expected_exotic_races = set(
        rn for rn, rd in pre_selections.get("races", {}).items()
        if rd.get("exotic")
    )
    missing_exotics = expected_exotic_races - parsed_exotic_races
    for race_num in sorted(missing_exotics):
        warnings.append(f"Race {race_num}: exotic not parsed")

    return warnings


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
            tracked_only = False
            no_bet_reason = None
            # Check tracked first — _BET_LINE's 300-char window can
            # reach the next pick's "Bet: $X" line and match the wrong one.
            tracked_m = _BET_TRACKED.search(after_text)
            bet_m = _BET_LINE.search(after_text)
            if tracked_m and (not bet_m or tracked_m.start() < bet_m.start()):
                bet_stake = 0.0
                bet_type = "place"  # preserve intended type for accuracy tracking
                tracked_only = True
                no_bet_reason = tracked_m.group(1).strip() if tracked_m.group(1) else None
            elif bet_m:
                bet_stake = float(bet_m.group(1))
                bet_type = _normalize_bet_type(bet_m.group(2))
                # E/W killed in pre-selections (Batch 1) — auto-convert to Place
                if bet_type == "each_way":
                    bet_type = "place"
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
            # Sanity: place odds must be less than win odds
            if place_odds and win_odds and place_odds > win_odds:
                logger.warning(
                    f"AI hallucinated place odds: {name.strip()} win=${win_odds} place=${place_odds}. "
                    f"Estimating place as (win-1)/3+1 = ${round((win_odds - 1) / 3 + 1, 2)}"
                )
                place_odds = round((win_odds - 1) / 3 + 1, 2)
            pick_dict = {
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
            }
            if tracked_only:
                pick_dict["tracked_only"] = True
                if no_bet_reason:
                    pick_dict["no_bet_reason"] = no_bet_reason
            picks.append(pick_dict)

        # Roughie (rank 4)
        roughie_m = _ROUGHIE.search(section)
        if roughie_m:
            after_text = section[roughie_m.end():roughie_m.end() + 300]
            tracked_only = False
            no_bet_reason = None
            tracked_m = _BET_TRACKED.search(after_text)
            bet_m = _BET_LINE.search(after_text)
            if tracked_m and (not bet_m or tracked_m.start() < bet_m.start()):
                bet_stake = 0.0
                bet_type = "place"
                tracked_only = True
                no_bet_reason = tracked_m.group(1).strip() if tracked_m.group(1) else None
            elif bet_m:
                bet_stake = float(bet_m.group(1))
                bet_type = _normalize_bet_type(bet_m.group(2))
                # E/W killed in pre-selections (Batch 1) — auto-convert to Place
                if bet_type == "each_way":
                    bet_type = "place"
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
            # Sanity: place odds must be less than win odds
            if place_odds and win_odds and place_odds > win_odds:
                logger.warning(
                    f"AI hallucinated place odds: {roughie_name.strip()} win=${win_odds} place=${place_odds}. "
                    f"Estimating place as (win-1)/3+1 = ${round((win_odds - 1) / 3 + 1, 2)}"
                )
                place_odds = round((win_odds - 1) / 3 + 1, 2)
            roughie_dict = {
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
            }
            if tracked_only:
                roughie_dict["tracked_only"] = True
                if no_bet_reason:
                    roughie_dict["no_bet_reason"] = no_bet_reason
            picks.append(roughie_dict)

        # Punty's Pick — mark the highlighted best-bet selection(s) or create exotic pick
        puntys_pick_m = _PUNTYS_PICK.search(section)
        exotic_pp_created = False  # Track if PP exotic was created to avoid duplicates
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
                exotic_pp_created = True
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

        # Exotic — try primary pattern (with Degenerate Exotic header) then fallback
        # Skip if Punty's Pick already created an exotic for this race
        exotic_m = None
        if not exotic_pp_created:
            exotic_m = _EXOTIC.search(section)
            if not exotic_m:
                exotic_m = _EXOTIC_FALLBACK.search(section)
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
        r"\*{0,2}SEQUENCE\s+LANES\*{0,2}[^\n]*\n(.*?)(?=\n###|\n\*{1,2}NUGGETS|\Z)",
        raw_content, re.DOTALL | re.IGNORECASE,
    )
    if not seq_section_m:
        # Try without the heading wrapper
        seq_section_m = re.search(
            r"((?:EARLY\s+QUADDIE|MAIN\s+QUADDIE|QUADDIE|BIG\s*6)\s*(?:\(|:)(?:Races?\s*)?R?\d+.*?)(?=\n###|\n\*{1,2}NUGGETS|\Z)",
            raw_content, re.DOTALL | re.IGNORECASE,
        )
    if not seq_section_m:
        return picks

    seq_text = seq_section_m.group(1) if seq_section_m else ""

    # Find each sequence block
    headers = list(_SEQ_HEADER.finditer(seq_text))
    # Track which sequence types have had their first (wagered) variant seen.
    # Only the first variant per sequence type is the actual bet — the rest
    # are tracked_only (displayed but not settled/staked).
    seen_seq_types: set[str] = set()

    for idx, hdr in enumerate(headers):
        seq_name = hdr.group(1).strip().upper()
        start_race = int(hdr.group(2) or hdr.group(4))
        end_race = int(hdr.group(3) or hdr.group(5))

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
            variant = vm.group(1).strip().lower().replace(" ", "_")
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

            # First variant per sequence type is the actual bet.
            # Subsequent variants (e.g. balanced, wide after skinny) are
            # tracked_only — displayed but not settled or staked.
            is_first_variant = seq_type not in seen_seq_types
            seen_seq_types.add(seq_type)

            pick_dict = {
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
                "exotic_stake": total_outlay if is_first_variant else 0.0,
                "estimated_return_pct": est_return_pct,
                "sequence_type": seq_type,
                "sequence_variant": variant,
                "sequence_legs": json.dumps(legs),
                "sequence_start_race": start_race,
                "multi_odds": None,
            }
            if not is_first_variant:
                pick_dict["tracked_only"] = True
                pick_dict["no_bet_reason"] = "Alternative variant — first variant is the wagered bet"
            picks.append(pick_dict)

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
