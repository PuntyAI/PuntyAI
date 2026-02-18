"""Pre-race change detection — scratchings, track changes, jockey/gear swaps."""

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class ChangeAlert:
    """A detected change that warrants a live update."""

    change_type: str  # "scratching", "track_condition", "jockey_change", "gear_change"
    meeting_id: str
    race_number: Optional[int] = None
    horse_name: Optional[str] = None
    saddlecloth: Optional[int] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    impacted_picks: list[dict] = field(default_factory=list)
    message: str = ""

    @property
    def dedup_key(self) -> str:
        """Unique key to prevent duplicate alerts."""
        if self.change_type == "track_condition":
            return f"track:{self.old_value}->{self.new_value}"
        if self.change_type == "weather":
            return f"weather:{self.message[:60]}"
        if self.change_type == "gear_change":
            return f"gear_change:R{self.race_number}"
        return f"{self.change_type}:R{self.race_number}:{self.horse_name}"


@dataclass
class RunnerSnapshot:
    """Snapshot of a runner's state before refresh."""

    race_number: int
    horse_name: str
    saddlecloth: int
    scratched: bool
    jockey: Optional[str]
    gear: Optional[str]
    gear_changes: Optional[str]


@dataclass
class MeetingSnapshot:
    """Pre-refresh snapshot of meeting + runner state."""

    track_condition: Optional[str]
    runners: dict[str, RunnerSnapshot] = field(default_factory=dict)  # key: "race_id:saddlecloth"


# ── Snapshot helpers ────────────────────────────────────────────────────────


async def take_snapshot(
    db: AsyncSession, meeting_id: str, upcoming_race_nums: list[int]
) -> MeetingSnapshot:
    """Capture current state of meeting + runners for upcoming races."""
    from punty.models.meeting import Meeting, Runner, Race

    meeting = await db.get(Meeting, meeting_id)
    snapshot = MeetingSnapshot(
        track_condition=meeting.track_condition if meeting else None,
    )

    for race_num in upcoming_race_nums:
        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(Runner.race_id == race_id)
        )
        for runner in result.scalars().all():
            key = f"{race_id}:{runner.saddlecloth}"
            snapshot.runners[key] = RunnerSnapshot(
                race_number=race_num,
                horse_name=runner.horse_name or "",
                saddlecloth=runner.saddlecloth or 0,
                scratched=bool(runner.scratched),
                jockey=runner.jockey,
                gear=runner.gear,
                gear_changes=runner.gear_changes,
            )

    return snapshot


# ── Scratching detection ───────────────────────────────────────────────────


async def detect_scratching_changes(
    db: AsyncSession,
    meeting_id: str,
    upcoming_race_nums: list[int],
    pre_snapshot: MeetingSnapshot,
) -> list[ChangeAlert]:
    """Find runners that are newly scratched since snapshot."""
    from punty.models.meeting import Runner

    alerts = []

    for race_num in upcoming_race_nums:
        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.scratched == True,
            )
        )
        for runner in result.scalars().all():
            key = f"{race_id}:{runner.saddlecloth}"
            prev = pre_snapshot.runners.get(key)
            if prev and prev.scratched:
                continue  # Already scratched before refresh

            horse = runner.horse_name or "Unknown"
            sc = runner.saddlecloth or 0

            picks = await find_impacted_picks(db, meeting_id, race_num, horse, sc)
            alt = await find_alternative(db, meeting_id, race_num, sc) if picks else None

            msg = compose_scratching_alert(
                horse_name=horse,
                race_number=race_num,
                impacted_picks=picks,
                alternative=alt,
            )

            alerts.append(ChangeAlert(
                change_type="scratching",
                meeting_id=meeting_id,
                race_number=race_num,
                horse_name=horse,
                saddlecloth=sc,
                old_value="active",
                new_value="scratched",
                impacted_picks=picks,
                message=msg,
            ))

    return alerts


# ── Track condition detection ──────────────────────────────────────────────


async def detect_track_condition_change(
    db: AsyncSession,
    meeting_id: str,
    pre_snapshot: MeetingSnapshot,
) -> Optional[ChangeAlert]:
    """Detect track condition change since snapshot."""
    from punty.models.meeting import Meeting

    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        return None

    old = pre_snapshot.track_condition
    new = meeting.track_condition

    if not old or not new or _base_condition(old) == _base_condition(new):
        return None

    msg = compose_track_alert(
        venue=meeting.venue or meeting_id,
        old_condition=old,
        new_condition=new,
    )

    return ChangeAlert(
        change_type="track_condition",
        meeting_id=meeting_id,
        old_value=old,
        new_value=new,
        message=msg,
    )


def _normalise_condition(cond: str) -> str:
    """Normalise track condition string for comparison.

    Handles format variations between sources:
    - "Good 4", "Good (4)", "good 4", "GOOD 4", "Good  4" all normalise to "good 4"
    """
    if not cond:
        return ""
    import re
    return re.sub(r"\s+", " ", cond.strip().lower().replace("(", "").replace(")", ""))


def _base_condition(cond: str) -> str:
    """Extract base condition word, ignoring numeric rating.

    Only alerts on genuine condition changes (Good→Soft), not rating fluctuations
    (Good 4→Good) caused by different sources reporting different detail levels.

    Examples: 'Good 4' -> 'good', 'Soft 7' -> 'soft', 'Heavy 8' -> 'heavy'
    """
    if not cond:
        return ""
    import re
    return re.sub(r"[\s\d()]+", " ", cond.strip().lower()).strip()


# ── Jockey / gear detection ───────────────────────────────────────────────


async def detect_jockey_gear_changes(
    db: AsyncSession,
    meeting_id: str,
    upcoming_race_nums: list[int],
    pre_snapshot: MeetingSnapshot,
) -> list[ChangeAlert]:
    """Find jockey or gear changes on runners that are in our picks."""
    from punty.models.meeting import Runner

    alerts = []
    # Collect gear changes per race for batching
    gear_changes_by_race: dict[int, list[dict]] = {}

    for race_num in upcoming_race_nums:
        race_id = f"{meeting_id}-r{race_num}"
        result = await db.execute(
            select(Runner).where(
                Runner.race_id == race_id,
                Runner.scratched == False,
            )
        )
        for runner in result.scalars().all():
            key = f"{race_id}:{runner.saddlecloth}"
            prev = pre_snapshot.runners.get(key)
            if not prev:
                continue

            horse = runner.horse_name or "Unknown"
            sc = runner.saddlecloth or 0

            # Check jockey change
            if (
                prev.jockey
                and runner.jockey
                and not _same_jockey(prev.jockey, runner.jockey)
            ):
                picks = await find_impacted_picks(db, meeting_id, race_num, horse, sc)
                if picks:
                    msg = compose_jockey_alert(
                        horse_name=horse,
                        race_number=race_num,
                        old_jockey=prev.jockey,
                        new_jockey=runner.jockey,
                        tip_rank=picks[0].get("tip_rank"),
                    )
                    alerts.append(ChangeAlert(
                        change_type="jockey_change",
                        meeting_id=meeting_id,
                        race_number=race_num,
                        horse_name=horse,
                        saddlecloth=sc,
                        old_value=prev.jockey,
                        new_value=runner.jockey,
                        impacted_picks=picks,
                        message=msg,
                    ))

            # Collect gear changes for batching per race
            if (
                runner.gear_changes
                and runner.gear_changes != prev.gear_changes
            ):
                picks = await find_impacted_picks(db, meeting_id, race_num, horse, sc)
                if picks:
                    if race_num not in gear_changes_by_race:
                        gear_changes_by_race[race_num] = []
                    gear_changes_by_race[race_num].append({
                        "horse_name": horse,
                        "saddlecloth": sc,
                        "gear_changes": runner.gear_changes,
                        "old_gear": prev.gear_changes,
                        "tip_rank": picks[0].get("tip_rank"),
                        "picks": picks,
                    })

    # Compose batched gear change alerts (one per race)
    for race_num, changes in gear_changes_by_race.items():
        msg = compose_gear_alert_batched(race_num, changes)
        # Use first horse's info for the alert metadata
        first = changes[0]
        alerts.append(ChangeAlert(
            change_type="gear_change",
            meeting_id=meeting_id,
            race_number=race_num,
            horse_name=first["horse_name"],
            saddlecloth=first["saddlecloth"],
            old_value=first["old_gear"],
            new_value=first["gear_changes"],
            impacted_picks=[p for c in changes for p in c["picks"]],
            message=msg,
        ))

    return alerts


def _normalise_name(name: str) -> str:
    """Normalise jockey name for comparison (strip dots, extra spaces)."""
    return name.strip().replace(".", "").lower()


def _clean_jockey_name(name: str) -> str:
    """Strip apprentice weight, titles, and punctuation from jockey name.

    Examples:
    - "Bailey Kinninmont(a2/52.5kg)" → "bailey kinninmont"
    - "Ms Sarah Field(a1.5/52.5kg)" → "sarah field"
    - "L.K.Cartwright" → "l k cartwright"
    - "C.Newitt" → "c newitt"
    - "Jace McMurray(a2/53kg), (late alt)" → "jace mcmurray"
    """
    import re
    # Strip parenthesised content (claim weights, late alt, etc.)
    name = re.sub(r"\([^)]*\)", "", name)
    # Replace dots and commas with spaces, normalise whitespace
    name = name.strip().replace(".", " ").replace(",", " ").lower()
    # Strip common title prefixes
    name = re.sub(r"^(ms|mr|mrs|miss|dr)\s+", "", name)
    return re.sub(r"\s+", " ", name).strip()


_SURNAME_PREFIXES = frozenset({"du", "de", "van", "von", "le", "la", "di", "el", "del", "den", "der", "dos", "das", "al"})


def _same_jockey(name_a: str, name_b: str) -> bool:
    """Check if two jockey name strings refer to the same person.

    Handles abbreviated vs full names, apprentice weights, title prefixes,
    and surname prefixes (Du Plessis, Le Boeuf, Van Der Westhuizen, etc.):
    - "Craig Newitt" vs "C.Newitt" → same
    - "Michael Dee" vs "M.J.Dee" → same
    - "Bailey Kinninmont(a2/52.5kg)" vs "B.R.Kinninmont" → same
    - "Ms Sarah Field(a1.5/52.5kg)" vs "S.Field" → same
    - "Luke Cartwright" vs "L.K.Cartwright" → same
    - "Mark Du Plessis" vs "M.R.du Plessis" → same
    - "Valentin Le Boeuf" vs "V.L.Boeuf" → same
    - "J. McDonald" vs "C. Williams" → different
    """
    a = _clean_jockey_name(name_a).split()
    b = _clean_jockey_name(name_b).split()
    if not a or not b:
        return False

    # Core surname = last token (always the actual family name)
    if a[-1] != b[-1]:
        return False

    # Given names = everything before last token, excluding surname prefixes.
    # Also strip single-char initials immediately before the surname if they
    # match a prefix initial (handles "V.L.Boeuf" where "L" = "Le").
    # Only strip from the last position to avoid eating real first-name initials.
    prefix_initials = {p[0] for p in _SURNAME_PREFIXES}
    a_given = list(a[:-1])
    b_given = list(b[:-1])
    # Strip full surname prefixes from anywhere
    a_first = [p for p in a_given if p not in _SURNAME_PREFIXES]
    b_first = [p for p in b_given if p not in _SURNAME_PREFIXES]
    # Strip trailing single-char prefix initial (right before surname)
    if a_first and len(a_first[-1]) == 1 and a_first[-1] in prefix_initials:
        a_first = a_first[:-1]
    if b_first and len(b_first[-1]) == 1 and b_first[-1] in prefix_initials:
        b_first = b_first[:-1]

    if not a_first or not b_first:
        return True  # Surname match with no given name to compare

    # First initial must match
    if a_first[0][0] != b_first[0][0]:
        return False

    # If abbreviated side has multiple initials, check they all match
    short, long = (a_first, b_first) if len(a_first) <= len(b_first) else (b_first, a_first)
    for i, part in enumerate(short):
        if i >= len(long):
            break
        if part[0] != long[i][0]:
            return False

    return True


# ── Pick impact analysis ──────────────────────────────────────────────────


async def find_impacted_picks(
    db: AsyncSession,
    meeting_id: str,
    race_number: int,
    horse_name: str,
    saddlecloth: int,
) -> list[dict]:
    """Find picks that reference this horse."""
    from punty.models.pick import Pick

    impacted = []

    # Selections + big3 for this race
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type.in_(["selection", "big3"]),
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        if _horse_matches(pick, horse_name, saddlecloth):
            impacted.append({
                "pick_id": pick.id,
                "pick_type": pick.pick_type,
                "tip_rank": pick.tip_rank,
                "bet_type": pick.bet_type,
                "bet_stake": pick.bet_stake,
            })

    # Exotics for this race
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type == "exotic",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        if pick.exotic_runners:
            try:
                runners = json.loads(pick.exotic_runners)
                if saddlecloth in runners:
                    impacted.append({
                        "pick_id": pick.id,
                        "pick_type": "exotic",
                        "exotic_type": pick.exotic_type,
                        "remaining_runners": len([r for r in runners if r != saddlecloth]),
                        "total_runners": len(runners),
                    })
            except (json.JSONDecodeError, TypeError):
                pass

    # Sequences (quaddie/big6) — check if this race is a leg
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "sequence",
            Pick.settled == False,
        )
    )
    for pick in result.scalars().all():
        if pick.sequence_legs and pick.sequence_start_race:
            try:
                legs = json.loads(pick.sequence_legs)
                leg_index = race_number - pick.sequence_start_race
                if 0 <= leg_index < len(legs):
                    leg = legs[leg_index]
                    if saddlecloth in leg:
                        impacted.append({
                            "pick_id": pick.id,
                            "pick_type": "sequence",
                            "sequence_type": pick.sequence_type,
                            "sequence_variant": pick.sequence_variant,
                            "leg_number": leg_index + 1,
                            "remaining_in_leg": len([r for r in leg if r != saddlecloth]),
                            "total_in_leg": len(leg),
                        })
            except (json.JSONDecodeError, TypeError):
                pass

    return impacted


def _horse_matches(pick, horse_name: str, saddlecloth: int) -> bool:
    """Check if a pick references this horse by name or saddlecloth."""
    if pick.saddlecloth and pick.saddlecloth == saddlecloth:
        return True
    if pick.horse_name and horse_name:
        return pick.horse_name.strip().lower() == horse_name.strip().lower()
    return False


# ── Alternative finder ────────────────────────────────────────────────────


async def find_alternative(
    db: AsyncSession,
    meeting_id: str,
    race_number: int,
    scratched_saddlecloth: int,
) -> Optional[dict]:
    """Find next-best runner to suggest as an alternative."""
    from punty.models.meeting import Runner
    from punty.models.pick import Pick

    race_id = f"{meeting_id}-r{race_number}"

    # Get existing pick saddlecloths for this race (to exclude)
    pick_result = await db.execute(
        select(Pick.saddlecloth).where(
            Pick.meeting_id == meeting_id,
            Pick.race_number == race_number,
            Pick.pick_type.in_(["selection", "big3"]),
        )
    )
    existing_saddlecloths = {row[0] for row in pick_result.all() if row[0]}

    # Get non-scratched runners sorted by odds
    runner_result = await db.execute(
        select(Runner).where(
            Runner.race_id == race_id,
            Runner.scratched == False,
            Runner.saddlecloth != scratched_saddlecloth,
            Runner.current_odds.isnot(None),
            Runner.current_odds > 0,
        ).order_by(Runner.current_odds.asc())
    )
    candidates = runner_result.scalars().all()

    # Prefer runners not already in our picks, on-pace
    for runner in candidates:
        if runner.saddlecloth in existing_saddlecloths:
            continue
        return {
            "horse_name": runner.horse_name,
            "saddlecloth": runner.saddlecloth,
            "odds": runner.current_odds,
            "speed_map_position": runner.speed_map_position,
        }

    # Fall back to any non-scratched runner not already picked
    for runner in candidates:
        if runner.saddlecloth not in existing_saddlecloths:
            return {
                "horse_name": runner.horse_name,
                "saddlecloth": runner.saddlecloth,
                "odds": runner.current_odds,
                "speed_map_position": runner.speed_map_position,
            }

    return None


# ── Message composition ───────────────────────────────────────────────────

# Punty-style scratching reactions
_SCRATCH_PHRASES = [
    "Well that's cooked.",
    "Of course.",
    "Brilliant timing.",
    "Typical.",
    "Pain.",
    "Righto then.",
]

_TRACK_UPGRADE_PHRASES = [
    "Good news for the dry trackers.",
    "Firming up nicely.",
    "Track's come good.",
]

_TRACK_DOWNGRADE_PHRASES = [
    "Getting sloppy out there.",
    "Mudlarks time.",
    "Track's gone.",
]


def compose_scratching_alert(
    horse_name: str,
    race_number: int,
    impacted_picks: list[dict],
    alternative: Optional[dict] = None,
) -> str:
    """Compose a scratching alert message (under 280 chars)."""
    # Determine impact description
    selection_picks = [p for p in impacted_picks if p["pick_type"] in ("selection", "big3")]
    exotic_picks = [p for p in impacted_picks if p["pick_type"] == "exotic"]
    seq_picks = [p for p in impacted_picks if p["pick_type"] == "sequence"]

    parts = []

    if selection_picks:
        rank = selection_picks[0].get("tip_rank")
        rank_str = f" (our #{rank} pick)" if rank else ""
        phrase = random.choice(_SCRATCH_PHRASES)
        parts.append(f"SCRATCHING: {horse_name}{rank_str} out of R{race_number}. {phrase}")
    elif exotic_picks or seq_picks:
        parts.append(f"SCRATCHING: {horse_name} out of R{race_number}.")
    else:
        parts.append(f"SCRATCHING: {horse_name} out of R{race_number}.")

    # Exotic impact
    for ep in exotic_picks:
        rem = ep.get("remaining_runners", 0)
        total = ep.get("total_runners", 0)
        etype = ep.get("exotic_type", "Exotic")
        parts.append(f"{etype} now {rem} of {total} runners.")

    # Sequence impact
    for sp in seq_picks:
        rem = sp.get("remaining_in_leg", 0)
        stype = sp.get("sequence_variant") or sp.get("sequence_type", "Sequence")
        leg = sp.get("leg_number", "?")
        parts.append(f"{stype.title()} Leg {leg} down to {rem} runner{'s' if rem != 1 else ''}.")

    # Alternative suggestion
    if alternative and selection_picks:
        alt_name = alternative["horse_name"]
        alt_odds = alternative.get("odds")
        alt_pos = alternative.get("speed_map_position")
        pos_str = f" ({alt_pos})" if alt_pos else ""
        odds_str = f" at ${alt_odds:.2f}" if alt_odds else ""
        parts.append(f"Next best: {alt_name}{odds_str}{pos_str}")

    msg = " ".join(parts)
    if len(msg) > 275:
        msg = msg[:272] + "..."
    return msg


def compose_track_alert(
    venue: str,
    old_condition: str,
    new_condition: str,
) -> str:
    """Compose a track condition change alert."""
    old_num = _extract_track_number(old_condition)
    new_num = _extract_track_number(new_condition)

    if old_num is not None and new_num is not None:
        upgraded = new_num < old_num  # Lower number = firmer/better
    else:
        upgraded = "good" in new_condition.lower() or "firm" in new_condition.lower()

    phrase = random.choice(_TRACK_UPGRADE_PHRASES if upgraded else _TRACK_DOWNGRADE_PHRASES)

    msg = f"TRACK UPDATE: {venue} {old_condition} \u2192 {new_condition}. {phrase}"

    if len(msg) > 275:
        msg = msg[:272] + "..."
    return msg


def compose_jockey_alert(
    horse_name: str,
    race_number: int,
    old_jockey: str,
    new_jockey: str,
    tip_rank: Optional[int] = None,
) -> str:
    """Compose a jockey change alert."""
    rank_str = f", our #{tip_rank} pick" if tip_rank else ""
    msg = f"JOCKEY CHANGE: {horse_name} (R{race_number}{rank_str}) \u2014 {old_jockey} off, {new_jockey} on"

    if len(msg) > 275:
        msg = msg[:272] + "..."
    return msg


def compose_gear_alert(
    horse_name: str,
    race_number: int,
    gear_changes: str,
    tip_rank: Optional[int] = None,
) -> str:
    """Compose a gear change alert for a single horse."""
    rank_str = f", our #{tip_rank} pick" if tip_rank else ""
    msg = f"GEAR CHANGE: {horse_name} (R{race_number}{rank_str}) \u2014 {gear_changes}"

    if len(msg) > 275:
        msg = msg[:272] + "..."
    return msg


def compose_gear_alert_batched(
    race_number: int,
    changes: list[dict],
) -> str:
    """Compose a batched gear change alert for multiple horses in one race."""
    if len(changes) == 1:
        c = changes[0]
        return compose_gear_alert(c["horse_name"], race_number, c["gear_changes"], c.get("tip_rank"))

    parts = [f"GEAR CHANGES R{race_number}:"]
    for c in changes:
        rank_str = f" (#{c['tip_rank']})" if c.get("tip_rank") else ""
        parts.append(f"{c['horse_name']}{rank_str} \u2014 {c['gear_changes']}")

    msg = " ".join(parts)
    if len(msg) > 275:
        msg = msg[:272] + "..."
    return msg


def _extract_track_number(condition: str) -> Optional[int]:
    """Extract numeric rating from track condition string (e.g., 'Soft 5' -> 5)."""
    import re
    m = re.search(r"(\d+)", condition)
    return int(m.group(1)) if m else None
