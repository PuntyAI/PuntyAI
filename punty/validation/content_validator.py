"""Post-generation content validation against probability data.

Validates parsed picks from AI-generated early mail content to catch
errors before approval. Returns warnings (advisory) and errors (blocking).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Validation thresholds
PUNTYS_PICK_MIN_PROB = 0.15      # 15% minimum for Punty's Pick
WIN_BET_MIN_PROB = 0.10          # 10% minimum for Win bets
STAKE_TOTAL_TARGET = 20.0        # $20 pool per race
STAKE_TOLERANCE = 3.0            # ±$3 tolerance
EXOTIC_MIN_RUNNERS = 2           # minimum runners for any exotic
TRIFECTA_RUNNERS = 3             # minimum for trifecta
FIRST4_RUNNERS = 4               # minimum for First4


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str          # "error" or "warning"
    race_number: int    # 0 = meeting-level
    message: str
    category: str       # "probability", "stake", "exotic", "sequence", "consistency"


@dataclass
class ValidationResult:
    """Complete validation result for an early mail."""

    issues: list[ValidationIssue] = field(default_factory=list)
    races_checked: int = 0
    picks_checked: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        if not self.issues:
            return f"Valid — {self.picks_checked} picks across {self.races_checked} races"
        errs = len(self.errors)
        warns = len(self.warnings)
        parts = []
        if errs:
            parts.append(f"{errs} error{'s' if errs > 1 else ''}")
        if warns:
            parts.append(f"{warns} warning{'s' if warns > 1 else ''}")
        return f"{', '.join(parts)} ({self.picks_checked} picks, {self.races_checked} races)"


def validate_content(
    parsed_picks: list[dict],
    race_data: dict[int, dict],
) -> ValidationResult:
    """Validate parsed picks against race probability data.

    Args:
        parsed_picks: List of pick dicts from parser.extract_all_picks()
            Each has: pick_type, race_number, saddlecloth, horse_name,
            bet_type, bet_stake, odds_at_tip, tip_rank, etc.
        race_data: Dict mapping race_number → dict with:
            - runners: list of runner dicts (saddlecloth, horse_name, _win_prob_raw, etc.)
            - field_size: int
            - pre_selections: optional RacePreSelections

    Returns:
        ValidationResult with all issues found.
    """
    result = ValidationResult()

    # Group picks by race
    picks_by_race: dict[int, list[dict]] = {}
    for pick in parsed_picks:
        rn = pick.get("race_number", 0)
        if rn not in picks_by_race:
            picks_by_race[rn] = []
        picks_by_race[rn].append(pick)

    result.races_checked = len(picks_by_race)
    result.picks_checked = len(parsed_picks)

    for race_num, race_picks in picks_by_race.items():
        rd = race_data.get(race_num, {})
        runners = rd.get("runners", [])
        runner_map = {r.get("saddlecloth"): r for r in runners if r.get("saddlecloth")}

        # Check selections
        selections = [p for p in race_picks if p.get("pick_type") == "selection"]
        _validate_selections(selections, runner_map, race_num, result)

        # Check exotics
        exotics = [p for p in race_picks if p.get("pick_type") == "exotic"]
        _validate_exotics(exotics, runner_map, race_num, result)

        # Check sequences
        sequences = [p for p in race_picks if p.get("pick_type") == "sequence"]
        _validate_sequences(sequences, race_data, race_num, result)

    # Check Punty's Pick across all races
    _validate_puntys_picks(parsed_picks, race_data, result)

    return result


def _validate_selections(
    selections: list[dict],
    runner_map: dict[int, dict],
    race_num: int,
    result: ValidationResult,
) -> None:
    """Validate selection picks for a single race."""
    # Check stake totals
    total_stake = 0.0
    has_win_bet = False

    for pick in selections:
        sc = pick.get("saddlecloth")
        bet_type = (pick.get("bet_type") or "").lower().strip()
        stake = pick.get("bet_stake", 0) or 0

        # Check runner exists
        if sc and sc not in runner_map:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=race_num,
                message=f"No.{sc} {pick.get('horse_name', '?')} not found in race field",
                category="consistency",
            ))
            continue

        runner = runner_map.get(sc, {})
        win_prob = runner.get("_win_prob_raw", 0)

        # Win bet probability check
        if bet_type in ("win", "saver win"):
            has_win_bet = True
            if win_prob and win_prob < WIN_BET_MIN_PROB:
                result.issues.append(ValidationIssue(
                    level="warning",
                    race_number=race_num,
                    message=(
                        f"Win bet on {pick.get('horse_name', '?')} (No.{sc}) "
                        f"with only {win_prob*100:.1f}% probability (min {WIN_BET_MIN_PROB*100:.0f}%)"
                    ),
                    category="probability",
                ))

        if bet_type == "each way":
            has_win_bet = True
            # Each Way costs double
            total_stake += stake * 2
        else:
            total_stake += stake

    # Check mandatory Win/Each Way bet
    if selections and not has_win_bet:
        result.issues.append(ValidationIssue(
            level="warning",
            race_number=race_num,
            message="No Win/Saver Win/Each Way bet — mandatory rule violated",
            category="consistency",
        ))

    # Check stake total
    if selections and total_stake > 0:
        if abs(total_stake - STAKE_TOTAL_TARGET) > STAKE_TOLERANCE:
            result.issues.append(ValidationIssue(
                level="warning",
                race_number=race_num,
                message=f"Stakes total ${total_stake:.2f} (target ${STAKE_TOTAL_TARGET:.0f} ± ${STAKE_TOLERANCE:.0f})",
                category="stake",
            ))


def _validate_exotics(
    exotics: list[dict],
    runner_map: dict[int, dict],
    race_num: int,
    result: ValidationResult,
) -> None:
    """Validate exotic bets for a race."""
    for pick in exotics:
        exotic_type = (pick.get("exotic_type") or "").lower()
        runners_raw = pick.get("exotic_runners") or []

        # Parse runners (could be list of ints or JSON string)
        runners = _parse_exotic_runners(runners_raw)

        if not runners:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=race_num,
                message=f"Exotic {pick.get('exotic_type', '?')} has no parseable runners",
                category="exotic",
            ))
            continue

        # Check minimum runners for type
        if "trifecta" in exotic_type and len(runners) < TRIFECTA_RUNNERS:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=race_num,
                message=f"Trifecta needs {TRIFECTA_RUNNERS}+ runners, got {len(runners)}",
                category="exotic",
            ))

        if "first" in exotic_type and "4" in exotic_type and len(runners) < FIRST4_RUNNERS:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=race_num,
                message=f"First4 needs {FIRST4_RUNNERS}+ runners, got {len(runners)}",
                category="exotic",
            ))

        if len(runners) < EXOTIC_MIN_RUNNERS:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=race_num,
                message=f"Exotic needs {EXOTIC_MIN_RUNNERS}+ runners, got {len(runners)}",
                category="exotic",
            ))

        # Check all runners exist in race
        for sc in runners:
            if sc not in runner_map:
                result.issues.append(ValidationIssue(
                    level="error",
                    race_number=race_num,
                    message=f"Exotic runner No.{sc} not found in R{race_num} field",
                    category="exotic",
                ))


def _validate_sequences(
    sequences: list[dict],
    race_data: dict[int, dict],
    race_num: int,
    result: ValidationResult,
) -> None:
    """Validate sequence bet legs."""
    for pick in sequences:
        legs = pick.get("sequence_legs") or []
        if not legs:
            continue

        # Parse legs
        if isinstance(legs, str):
            try:
                import json
                legs = json.loads(legs)
            except (ValueError, TypeError):
                continue

        if not isinstance(legs, list):
            continue

        start_race = pick.get("sequence_start_race", 0)
        for i, leg in enumerate(legs):
            leg_race = start_race + i
            rd = race_data.get(leg_race, {})
            runners = rd.get("runners", [])
            runner_scs = {r.get("saddlecloth") for r in runners if r.get("saddlecloth")}

            if not isinstance(leg, list):
                continue

            for sc in leg:
                if isinstance(sc, int) and runner_scs and sc not in runner_scs:
                    result.issues.append(ValidationIssue(
                        level="error",
                        race_number=leg_race,
                        message=f"Sequence leg runner No.{sc} not in R{leg_race} field",
                        category="sequence",
                    ))

        # Validate combo maths
        _validate_combo_maths(pick, result)


def _validate_combo_maths(pick: dict, result: ValidationResult) -> None:
    """Validate that combos × unit = total outlay for sequences."""
    legs = pick.get("sequence_legs") or []
    if isinstance(legs, str):
        try:
            import json
            legs = json.loads(legs)
        except (ValueError, TypeError):
            return

    if not isinstance(legs, list) or not legs:
        return

    # Calculate expected combos
    expected_combos = 1
    for leg in legs:
        if isinstance(leg, list):
            expected_combos *= max(1, len(leg))

    # The pick should have combo count — check if it matches
    stated_combos = pick.get("sequence_combos")
    if stated_combos and isinstance(stated_combos, (int, float)):
        stated_combos = int(stated_combos)
        if stated_combos != expected_combos:
            result.issues.append(ValidationIssue(
                level="error",
                race_number=pick.get("sequence_start_race", 0),
                message=(
                    f"Combo maths error in {pick.get('sequence_variant', '?')}: "
                    f"stated {stated_combos} combos but legs give {expected_combos}"
                ),
                category="sequence",
            ))


def _validate_puntys_picks(
    all_picks: list[dict],
    race_data: dict[int, dict],
    result: ValidationResult,
) -> None:
    """Validate Punty's Pick selections have sufficient probability."""
    puntys = [p for p in all_picks
              if p.get("pick_type") == "selection" and p.get("tip_rank") == 0]

    for pick in puntys:
        sc = pick.get("saddlecloth")
        rn = pick.get("race_number", 0)
        rd = race_data.get(rn, {})
        runners = rd.get("runners", [])
        runner_map = {r.get("saddlecloth"): r for r in runners if r.get("saddlecloth")}

        runner = runner_map.get(sc, {})
        win_prob = runner.get("_win_prob_raw", 0)

        if win_prob and win_prob < PUNTYS_PICK_MIN_PROB:
            result.issues.append(ValidationIssue(
                level="warning",
                race_number=rn,
                message=(
                    f"Punty's Pick {pick.get('horse_name', '?')} (No.{sc}) "
                    f"has only {win_prob*100:.1f}% probability "
                    f"(min {PUNTYS_PICK_MIN_PROB*100:.0f}%)"
                ),
                category="probability",
            ))


def _parse_exotic_runners(runners_raw: Any) -> list[int]:
    """Parse exotic runners from various formats."""
    if isinstance(runners_raw, list):
        result = []
        for r in runners_raw:
            if isinstance(r, int):
                result.append(r)
            elif isinstance(r, str) and r.strip().isdigit():
                result.append(int(r.strip()))
        return result

    if isinstance(runners_raw, str):
        try:
            import json
            parsed = json.loads(runners_raw)
            if isinstance(parsed, list):
                return [int(r) for r in parsed if str(r).strip().isdigit()]
        except (ValueError, TypeError):
            pass
        # Try comma/slash separated
        nums = re.findall(r'\d+', runners_raw)
        return [int(n) for n in nums]

    return []


async def validate_early_mail_probability(
    content_raw: str,
    meeting_id: str,
    db,
) -> ValidationResult:
    """Full validation of early mail content against DB data.

    This is the async entry point that loads race data from the database
    and runs the validator. Call this from the approval flow.
    """
    from punty.results.parser import extract_all_picks
    from punty.models.meeting import Meeting, Race, Runner
    from punty.context.builder import ContextBuilder
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    # Parse picks from content
    parsed_picks = extract_all_picks(content_raw, meeting_id)

    # Load race data with probabilities
    result = await db.execute(
        select(Meeting)
        .where(Meeting.id == meeting_id)
        .options(selectinload(Meeting.races).selectinload(Race.runners))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        vr = ValidationResult()
        vr.issues.append(ValidationIssue(
            level="error", race_number=0,
            message=f"Meeting {meeting_id} not found",
            category="consistency",
        ))
        return vr

    # Build race data dict from runner DB data
    race_data: dict[int, dict] = {}
    for race in meeting.races:
        runners_list = []
        for runner in race.runners:
            if runner.scratched:
                continue
            runners_list.append({
                "saddlecloth": runner.saddlecloth,
                "horse_name": runner.horse_name,
                "_win_prob_raw": 0,  # probabilities not stored on DB runner
                "current_odds": runner.current_odds,
            })
        race_data[race.race_number] = {
            "runners": runners_list,
            "field_size": len(runners_list),
        }

    return validate_content(parsed_picks, race_data)
