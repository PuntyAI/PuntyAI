"""Race assessment generator for learning from predictions.

After races settle, this module generates structured LLM assessments
comparing predictions vs reality, extracting learnings for future RAG retrieval.
"""

import json
import logging
from typing import Any, Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.config import melb_now_naive
from punty.memory.models import RaceAssessment, RaceMemory
from punty.memory.embeddings import EmbeddingService
from punty.models.meeting import Meeting, Race, Runner
from punty.models.pick import Pick

logger = logging.getLogger(__name__)

# Track to state mapping for Australian tracks
TRACK_STATE_MAP = {
    # Victoria
    "flemington": "VIC", "caulfield": "VIC", "moonee valley": "VIC", "sandown": "VIC",
    "pakenham": "VIC", "cranbourne": "VIC", "mornington": "VIC", "ballarat": "VIC",
    "bendigo": "VIC", "geelong": "VIC", "sale": "VIC", "warrnambool": "VIC",
    "wangaratta": "VIC", "stawell": "VIC", "hamilton": "VIC", "kilmore": "VIC",
    "seymour": "VIC", "echuca": "VIC", "swan hill": "VIC", "mildura": "VIC",
    "bairnsdale": "VIC", "stony creek": "VIC", "tatura": "VIC", "kyneton": "VIC",
    "wodonga": "VIC", "ararat": "VIC", "horsham": "VIC", "moe": "VIC",
    "donald": "VIC", "avoca": "VIC", "edenhope": "VIC", "terang": "VIC",
    # NSW
    "randwick": "NSW", "rosehill": "NSW", "canterbury": "NSW", "warwick farm": "NSW",
    "newcastle": "NSW", "kembla grange": "NSW", "gosford": "NSW", "wyong": "NSW",
    "hawkesbury": "NSW", "scone": "NSW", "tamworth": "NSW", "muswellbrook": "NSW",
    "dubbo": "NSW", "bathurst": "NSW", "orange": "NSW", "goulburn": "NSW",
    "wagga": "NSW", "albury": "NSW", "canberra": "NSW", "queanbeyan": "NSW",
    "grafton": "NSW", "ballina": "NSW", "lismore": "NSW", "coffs harbour": "NSW",
    "port macquarie": "NSW", "taree": "NSW", "moree": "NSW", "armidale": "NSW",
    # Queensland
    "eagle farm": "QLD", "doomben": "QLD", "gold coast": "QLD", "sunshine coast": "QLD",
    "ipswich": "QLD", "toowoomba": "QLD", "rockhampton": "QLD", "mackay": "QLD",
    "townsville": "QLD", "cairns": "QLD", "beaudesert": "QLD", "kilcoy": "QLD",
    # South Australia
    "morphettville": "SA", "gawler": "SA", "murray bridge": "SA", "mount gambier": "SA",
    "strathalbyn": "SA", "balaklava": "SA", "port augusta": "SA", "port lincoln": "SA",
    # Western Australia
    "ascot": "WA", "belmont": "WA", "pinjarra": "WA", "bunbury": "WA",
    "northam": "WA", "kalgoorlie": "WA", "geraldton": "WA", "albany": "WA",
    # Tasmania
    "hobart": "TAS", "launceston": "TAS", "devonport": "TAS",
    # NT
    "darwin": "NT", "alice springs": "NT",
}


def _get_state_from_track(track: str) -> Optional[str]:
    """Derive state from track name."""
    if not track:
        return None
    track_lower = track.lower().strip()

    # Direct match first
    if track_lower in TRACK_STATE_MAP:
        return TRACK_STATE_MAP[track_lower]

    # Try partial match (e.g., "Southside Pakenham" -> "pakenham")
    for known_track, state in TRACK_STATE_MAP.items():
        if known_track in track_lower or track_lower in known_track:
            return state

    return None


def _get_sex_restriction(race_name: Optional[str], runners: list[dict]) -> Optional[str]:
    """Derive sex restriction from race name or runner composition."""
    name_lower = (race_name or "").lower()

    # Check race name for explicit restrictions
    if "fillies" in name_lower and "mares" in name_lower:
        return "F&M"
    if "fillies" in name_lower:
        return "Fillies"
    if "mares" in name_lower:
        return "Mares"
    if "colts" in name_lower and "geldings" in name_lower:
        return "C&G"
    if "colts" in name_lower:
        return "Colts"

    # Check runner composition if available
    if runners:
        sexes = set()
        for r in runners:
            sex = r.get("horse_sex", "").lower()
            if sex:
                sexes.add(sex)

        # All fillies/mares
        if sexes and all(s in ("f", "m", "filly", "mare") for s in sexes):
            return "F&M"
        # All colts/geldings/horses/ridglings
        if sexes and all(s in ("c", "g", "h", "r", "colt", "gelding", "horse", "ridgling") for s in sexes):
            return "C&G"

    return None  # Open/Mixed


def _format_runner_for_assessment(runner: Runner) -> dict[str, Any]:
    """Format a runner with ALL available data points for assessment."""
    return {
        "horse_name": runner.horse_name,
        "saddlecloth": runner.saddlecloth,
        "barrier": runner.barrier,
        "weight": runner.weight,
        "jockey": runner.jockey,
        "trainer": runner.trainer,
        "trainer_location": runner.trainer_location,
        "form": runner.form,
        "last_five": runner.last_five,
        "career_record": runner.career_record,
        # Horse details
        "horse_age": runner.horse_age,
        "horse_sex": runner.horse_sex,
        "sire": runner.sire,
        "dam": runner.dam,
        "dam_sire": runner.dam_sire,
        "career_prize_money": runner.career_prize_money,
        "days_since_last_run": runner.days_since_last_run,
        "handicap_rating": runner.handicap_rating,
        "speed_value": runner.speed_value,
        # Speed/Pace data
        "speed_map_position": runner.speed_map_position,
        "pf_speed_rank": runner.pf_speed_rank,
        "pf_settle": runner.pf_settle,
        "pf_map_factor": runner.pf_map_factor,
        "pf_jockey_factor": runner.pf_jockey_factor,
        # Odds data
        "opening_odds": runner.opening_odds,
        "current_odds": runner.current_odds,
        "place_odds": runner.place_odds,
        "odds_tab": runner.odds_tab,
        "odds_sportsbet": runner.odds_sportsbet,
        "odds_bet365": runner.odds_bet365,
        "odds_ladbrokes": runner.odds_ladbrokes,
        "odds_betfair": runner.odds_betfair,
        "odds_flucs": runner.odds_flucs,
        # Stats
        "track_dist_stats": runner.track_dist_stats,
        "track_stats": runner.track_stats,
        "distance_stats": runner.distance_stats,
        "first_up_stats": runner.first_up_stats,
        "second_up_stats": runner.second_up_stats,
        "good_track_stats": runner.good_track_stats,
        "soft_track_stats": runner.soft_track_stats,
        "heavy_track_stats": runner.heavy_track_stats,
        "jockey_stats": runner.jockey_stats,
        "trainer_stats": runner.trainer_stats,
        "class_stats": runner.class_stats,
        # Gear & comments
        "gear": runner.gear,
        "gear_changes": runner.gear_changes,
        "stewards_comment": runner.stewards_comment,
        "comment_long": runner.comment_long,
        "comment_short": runner.comment_short,
        # Result
        "finish_position": runner.finish_position,
        "result_margin": runner.result_margin,
        "starting_price": runner.starting_price,
        "win_dividend": runner.win_dividend,
        "place_dividend": runner.place_dividend,
        "sectional_400": runner.sectional_400,
        "sectional_800": runner.sectional_800,
        "scratched": runner.scratched,
        "scratching_reason": runner.scratching_reason,
    }


def _format_runners_table(runners: list[dict]) -> str:
    """Format runners as a readable table for the LLM."""
    lines = []
    for r in sorted(runners, key=lambda x: x.get("finish_position") or 99):
        # Skip scratched runners in final position summary
        if r.get("scratched"):
            continue
        pos = r.get("finish_position", "?")
        margin = r.get("result_margin", "")
        name = r.get("horse_name", "Unknown")
        barrier = r.get("barrier", "?")
        jockey = r.get("jockey", "")
        sp = r.get("starting_price") or r.get("current_odds") or "?"
        speed_pos = r.get("speed_map_position", "unknown")

        lines.append(f"{pos}. {name} (Bar {barrier}) - {jockey} - SP ${sp} - Pace: {speed_pos} {margin}")
    return "\n".join(lines)


def _format_predictions(picks: list[Pick]) -> str:
    """Format our predictions for the review."""
    if not picks:
        return "No selections were made for this race."

    lines = []
    for p in sorted(picks, key=lambda x: x.tip_rank or 99):
        rank_label = {1: "TOP PICK", 2: "2ND", 3: "3RD", 4: "ROUGHIE"}.get(p.tip_rank, f"#{p.tip_rank}")
        bet = p.bet_type or "win"
        odds = p.odds_at_tip or 0
        stake = p.bet_stake or 0
        lines.append(f"- {rank_label}: {p.horse_name} (No.{p.saddlecloth}) @ ${odds:.2f} - ${stake:.0f} {bet}")
    return "\n".join(lines)


def _format_prediction_results(picks: list[Pick], runners: list[dict]) -> str:
    """Format how our predictions actually performed."""
    if not picks:
        return "No selections to evaluate."

    lines = []
    total_pnl = 0.0

    for p in sorted(picks, key=lambda x: x.tip_rank or 99):
        # Find the runner's result
        runner = next((r for r in runners if r.get("saddlecloth") == p.saddlecloth), None)
        if not runner:
            lines.append(f"- {p.horse_name}: Could not find result")
            continue

        pos = runner.get("finish_position", "?")
        margin = runner.get("result_margin", "")
        sp = runner.get("starting_price") or runner.get("current_odds") or "?"
        pnl = p.pnl or 0
        total_pnl += pnl

        hit_marker = "✓" if p.hit else "✗"
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"${pnl:.2f}"
        lines.append(f"- {p.horse_name}: Finished {pos} {margin} (SP ${sp}) {hit_marker} P&L: {pnl_str}")

    lines.append(f"\nTotal Race P&L: ${total_pnl:.2f}")
    return "\n".join(lines)


def _format_sectional_times(sectional_data: Optional[dict]) -> str:
    """Format sectional times data for the prompt."""
    if not sectional_data or not sectional_data.get("horses"):
        return "Sectional times not available for this race."

    lines = ["### Post-Race Sectional Times (Actual Running)"]
    lines.append("Position and time at each checkpoint during the race:\n")

    for horse in sorted(sectional_data["horses"], key=lambda x: x.get("final_position") or 99):
        name = horse.get("horse_name", "Unknown")
        saddle = horse.get("saddlecloth", "?")
        pos = horse.get("final_position_abbr") or horse.get("final_position", "?")
        race_time = horse.get("race_time", "?")
        margin = horse.get("beaten_margin", 0)
        comment = horse.get("comment", "")

        lines.append(f"**{saddle}. {name}** - {pos} (Time: {race_time}, Margin: {margin}L)")

        # Show position progression through race
        sectionals = horse.get("sectional_times", [])
        if sectionals:
            positions = []
            for s in sectionals:
                dist = s.get("distance", "")
                sect_pos = s.get("position", "?")
                sect_time = s.get("time", "?")
                positions.append(f"{dist}: P{sect_pos} ({sect_time}s)")
            lines.append(f"  Positions: {' → '.join(positions)}")

        # Show split times (time between checkpoints)
        splits = horse.get("split_times", [])
        if splits:
            split_strs = []
            for s in splits:
                dist = s.get("distance", "")
                split_time = s.get("time", "?")
                split_strs.append(f"{dist}: {split_time}s")
            lines.append(f"  Splits: {' | '.join(split_strs)}")

        # Include the generated comment if available
        if comment:
            lines.append(f"  Comment: {comment}")

        lines.append("")

    return "\n".join(lines)


def _build_assessment_prompt(
    meeting: Meeting,
    race: Race,
    runners: list[dict],
    picks: list[Pick],
    sectional_data: Optional[dict] = None,
) -> str:
    """Build the comprehensive post-race assessment prompt."""

    # Format runner data with ALL fields
    runner_details = []
    for r in sorted(runners, key=lambda x: x.get("saddlecloth") or 99):
        if r.get("scratched"):
            continue
        details = [f"**{r['saddlecloth']}. {r['horse_name']}**"]

        # Pre-race profile
        details.append(f"  Barrier: {r.get('barrier')} | Weight: {r.get('weight')}kg | Jockey: {r.get('jockey')} | Trainer: {r.get('trainer')}")

        if r.get("horse_age") or r.get("horse_sex"):
            details.append(f"  Age/Sex: {r.get('horse_age') or '?'}yo {r.get('horse_sex') or ''}")

        if r.get("sire"):
            details.append(f"  Breeding: {r.get('sire')} x {r.get('dam')} ({r.get('dam_sire')})")

        # Form
        if r.get("form") or r.get("last_five"):
            details.append(f"  Form: {r.get('form') or ''} | Last 5: {r.get('last_five') or ''}")

        if r.get("days_since_last_run") is not None:
            details.append(f"  Days since last run: {r.get('days_since_last_run')}")

        # Stats
        stats = []
        if r.get("track_dist_stats"):
            stats.append(f"Track/Dist: {r.get('track_dist_stats')}")
        if r.get("track_stats"):
            stats.append(f"Track: {r.get('track_stats')}")
        if r.get("distance_stats"):
            stats.append(f"Distance: {r.get('distance_stats')}")
        if r.get("first_up_stats"):
            stats.append(f"1st up: {r.get('first_up_stats')}")
        if r.get("second_up_stats"):
            stats.append(f"2nd up: {r.get('second_up_stats')}")

        # Track condition stats
        track_cond = meeting.track_condition or ""
        if "heavy" in track_cond.lower() and r.get("heavy_track_stats"):
            stats.append(f"Heavy: {r.get('heavy_track_stats')}")
        elif "soft" in track_cond.lower() and r.get("soft_track_stats"):
            stats.append(f"Soft: {r.get('soft_track_stats')}")
        elif r.get("good_track_stats"):
            stats.append(f"Good: {r.get('good_track_stats')}")

        if r.get("jockey_stats"):
            stats.append(f"Jockey combo: {r.get('jockey_stats')}")
        if r.get("trainer_stats"):
            stats.append(f"Trainer: {r.get('trainer_stats')}")
        if r.get("class_stats"):
            stats.append(f"Class: {r.get('class_stats')}")

        if stats:
            details.append(f"  Stats: {' | '.join(stats)}")

        # Speed/Pace
        pace_info = []
        if r.get("speed_map_position"):
            pace_info.append(f"Map: {r.get('speed_map_position')}")
        if r.get("pf_speed_rank"):
            pace_info.append(f"Speed Rank: {r.get('pf_speed_rank')}")
        if r.get("pf_settle"):
            pace_info.append(f"Settle: {r.get('pf_settle')}")
        if r.get("pf_map_factor"):
            pace_info.append(f"Map Factor: {r.get('pf_map_factor')}")
        if r.get("pf_jockey_factor"):
            pace_info.append(f"Jockey Factor: {r.get('pf_jockey_factor')}")

        if pace_info:
            details.append(f"  Pace: {' | '.join(pace_info)}")

        # Odds
        odds_info = [f"Open: ${r.get('opening_odds') or '?'}", f"Final: ${r.get('current_odds') or '?'}"]
        if r.get("odds_flucs"):
            odds_info.append(f"Flucs: {r.get('odds_flucs')}")
        details.append(f"  Odds: {' | '.join(odds_info)}")

        # Gear
        if r.get("gear"):
            details.append(f"  Gear: {r.get('gear')}")
        if r.get("gear_changes"):
            details.append(f"  Gear Changes: {r.get('gear_changes')}")

        # Comments
        if r.get("comment_short"):
            details.append(f"  Comment: {r.get('comment_short')}")

        # Result
        details.append(f"  **RESULT**: Finished {r.get('finish_position')} {r.get('result_margin') or ''}")
        if r.get("sectional_800") or r.get("sectional_400"):
            details.append(f"  Sectionals: 800m {r.get('sectional_800') or '?'} | 400m {r.get('sectional_400') or '?'}")

        runner_details.append("\n".join(details))

    prompt = f"""You are reviewing a completed horse race to assess prediction accuracy and extract learnings for future improvement.

## Meeting & Race Details
- **Venue**: {meeting.venue}
- **Date**: {meeting.date}
- **Track Condition**: {meeting.track_condition or 'Unknown'}
- **Weather**: {meeting.weather or meeting.weather_condition or 'Unknown'} | Temp: {meeting.weather_temp or '?'}°C | Wind: {meeting.weather_wind_speed or '?'}km/h {meeting.weather_wind_dir or ''}
- **Rail Position**: {meeting.rail_position or 'True'}
- **Rail Bias Comment**: {meeting.rail_bias_comment or 'None'}
- **Penetrometer**: {meeting.penetrometer or 'N/A'}

## Race Details
- **Race {race.race_number}**: {race.name}
- **Distance**: {race.distance}m
- **Class**: {race.class_ or 'Unknown'}
- **Prize Money**: ${race.prize_money or 0:,}
- **Race Type**: {race.race_type or 'Flat'}
- **Age Restriction**: {race.age_restriction or 'Open'}
- **Weight Type**: {race.weight_type or 'Handicap'}
- **Field Size**: {race.field_size or len([r for r in runners if not r.get('scratched')])}
- **Winning Time**: {race.winning_time or 'N/A'}

## Our Predictions
{_format_predictions(picks)}

## Prediction Results
{_format_prediction_results(picks, runners)}

## Full Result
{_format_runners_table(runners)}

## Post-Race Sectional Times
{_format_sectional_times(sectional_data)}

## Detailed Runner Profiles (Pre-Race Data + Result)
{chr(10).join(runner_details)}

---

## Your Analysis Task

IMPORTANT: Compare the actual sectional running (positions at each checkpoint) with our pre-race speed map predictions.
This tells us if horses ran where we expected them to, and how the pace unfolded.

Review what happened vs what we predicted. Be honest and analytical - we want to learn.

Respond with a JSON object containing:

```json
{{
    "prediction_accuracy": {{
        "top_pick_result": "Where our top pick finished and why (be specific about what happened in the race)",
        "value_assessment": "Did our value picks deliver? Were the odds we got fair value or not?",
        "pace_prediction_accuracy": "Did the pace unfold as expected based on speed maps? What actually happened?",
        "key_misses": "What did we get wrong and why? Be specific about which factors we misjudged."
    }},
    "decisive_factors": {{
        "what_won_the_race": "The key factor(s) that determined the winner (barrier/pace/class/fitness/jockey/track position/wet form/etc)",
        "undervalued_factors": ["List factors we should have weighted MORE heavily"],
        "overvalued_factors": ["List factors we weighted too heavily that didn't matter"]
    }},
    "sectional_analysis": {{
        "speed_map_accuracy": "How accurate was our speed map? Did horses settle where predicted?",
        "pace_scenario": "How did the pace unfold? (e.g., slow/fast early, sprint home, even tempo)",
        "key_moves": "Which horses made significant moves through the race? When and where?",
        "winner_run_style": "How did the winner run the race? (led, stalked, came from back, etc.)"
    }},
    "track_learnings": {{
        "track": "{meeting.venue}",
        "distance": {race.distance},
        "going": "{meeting.track_condition or 'Unknown'}",
        "observed_bias": "Any rail/track bias observed in how positions played out",
        "pace_dynamics": "How pace played out at this track/distance/going combination",
        "barrier_impact": "How much did barriers matter in this race?"
    }},
    "market_analysis": {{
        "market_accuracy": "How well did the market (odds) predict the result?",
        "overlays_identified": "Were there genuine overlays we found or missed?",
        "market_moves_significance": "Did any significant market moves prove meaningful?"
    }},
    "improvement_recommendations": {{
        "what_worked": "Specific aspects of our analysis that were accurate",
        "what_failed": "Specific aspects that were wrong - be honest",
        "actionable_recommendation": "ONE specific, actionable change to improve predictions for similar races in future"
    }},
    "key_learnings_summary": "1-2 sentence summary capturing THE most important takeaway for future {meeting.venue} {race.distance}m races in {meeting.track_condition or 'similar'} conditions"
}}
```

Be specific and analytical. Reference actual runner data when explaining what happened.
Focus on learnings that will be useful for future race predictions.
"""
    return prompt


async def generate_race_assessment(
    db: AsyncSession,
    race_id: str,
    api_key: Optional[str] = None,
) -> Optional[RaceAssessment]:
    """Generate a post-race assessment for learning.

    This is called after a race settles to create a structured
    assessment of how our predictions performed.

    Args:
        db: Database session
        race_id: The race ID (e.g., "sale-2026-02-06-r1")
        api_key: OpenAI API key (if not provided, will fetch from settings)

    Returns:
        The created RaceAssessment, or None if assessment couldn't be generated
    """
    # Check if assessment already exists
    existing = await db.execute(
        select(RaceAssessment).where(RaceAssessment.race_id == race_id)
    )
    if existing.scalar_one_or_none():
        logger.info(f"Assessment already exists for {race_id}")
        return None

    # Fetch race with runners
    race_result = await db.execute(
        select(Race)
        .options(selectinload(Race.runners))
        .where(Race.id == race_id)
    )
    race = race_result.scalar_one_or_none()
    if not race:
        logger.warning(f"Race not found: {race_id}")
        return None

    # Fetch meeting
    meeting_result = await db.execute(
        select(Meeting).where(Meeting.id == race.meeting_id)
    )
    meeting = meeting_result.scalar_one_or_none()
    if not meeting:
        logger.warning(f"Meeting not found: {race.meeting_id}")
        return None

    # Fetch our picks for this race
    picks_result = await db.execute(
        select(Pick).where(
            and_(
                Pick.meeting_id == meeting.id,
                Pick.race_number == race.race_number,
                Pick.pick_type == "selection",
                Pick.settled == True,
            )
        )
    )
    picks = list(picks_result.scalars().all())

    if not picks:
        logger.info(f"No settled picks for {race_id}, skipping assessment")
        return None

    # Format runner data
    runners = [_format_runner_for_assessment(r) for r in race.runners]

    # Parse sectional times if available
    sectional_data = None
    if race.sectional_times:
        try:
            sectional_data = json.loads(race.sectional_times)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build prompt
    prompt = _build_assessment_prompt(meeting, race, runners, picks, sectional_data)

    # Get API key
    if not api_key:
        from punty.models.settings import get_api_key
        api_key = await get_api_key(db, "openai_api_key")

    if not api_key:
        logger.error("No OpenAI API key available for assessment generation")
        return None

    # Call OpenAI
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temp for more consistent analysis
        )

        assessment_text = response.choices[0].message.content
        assessment_data = json.loads(assessment_text)

    except Exception as e:
        logger.error(f"Failed to generate assessment for {race_id}: {e}")
        return None

    # Extract key learnings for embedding
    key_learnings = assessment_data.get("key_learnings_summary", "")
    if not key_learnings:
        # Fallback: construct from other fields
        track_info = assessment_data.get("track_learnings", {})
        rec = assessment_data.get("improvement_recommendations", {})
        key_learnings = f"{track_info.get('observed_bias', '')} {rec.get('actionable_recommendation', '')}"

    # Calculate metrics
    top_pick = next((p for p in picks if p.tip_rank == 1), None)
    top_pick_hit = top_pick.hit if top_pick else None
    any_pick_hit = any(p.hit for p in picks if p.hit is not None)
    total_pnl = sum(p.pnl or 0 for p in picks)

    # Derive additional fields
    state = _get_state_from_track(meeting.venue)
    sex_restriction = _get_sex_restriction(race.name, runners)
    field_size = len([r for r in runners if not r.get("scratched")])

    # Create assessment record
    assessment = RaceAssessment(
        race_id=race_id,
        meeting_id=meeting.id,
        race_number=race.race_number,
        track=meeting.venue,
        distance=race.distance,
        race_class=race.class_ or "Unknown",
        going=meeting.track_condition or "Unknown",
        rail_position=meeting.rail_position,
        # New fields for better matching
        age_restriction=race.age_restriction,
        sex_restriction=sex_restriction,
        weight_type=race.weight_type,
        field_size=field_size,
        prize_money=race.prize_money,
        penetrometer=meeting.penetrometer,
        state=state,
        weather=meeting.weather_condition or meeting.weather,
        temperature=meeting.weather_temp,
        # Metrics
        key_learnings=key_learnings,
        top_pick_hit=top_pick_hit,
        any_pick_hit=any_pick_hit,
        total_pnl=total_pnl,
        created_at=melb_now_naive(),
    )
    assessment.assessment = assessment_data

    # Generate embedding for similarity search
    try:
        embedding_service = EmbeddingService(api_key=api_key)
        # Create rich text for embedding with all relevant race characteristics
        embed_parts = [
            meeting.venue,
            f"{race.distance}m",
            race.class_ or "",
            meeting.track_condition or "",
        ]
        if race.age_restriction:
            embed_parts.append(race.age_restriction)
        if sex_restriction:
            embed_parts.append(sex_restriction)
        if race.weight_type:
            embed_parts.append(race.weight_type)
        if state:
            embed_parts.append(state)
        if meeting.weather_condition or meeting.weather:
            embed_parts.append(meeting.weather_condition or meeting.weather)
        embed_parts.append(key_learnings)

        embed_text = " ".join(filter(None, embed_parts))
        embedding = await embedding_service.get_embedding(embed_text)
        if embedding:
            assessment.embedding = embedding
    except Exception as e:
        logger.warning(f"Failed to generate embedding for assessment: {e}")

    db.add(assessment)
    await db.commit()

    logger.info(f"Generated assessment for {race_id}: top_pick_hit={top_pick_hit}, pnl={total_pnl:.2f}")
    return assessment


async def retrieve_assessment_context(
    db: AsyncSession,
    track: str,
    distance: int,
    going: str,
    race_class: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    # New optional parameters for better matching
    age_restriction: Optional[str] = None,
    sex_restriction: Optional[str] = None,
    weight_type: Optional[str] = None,
    field_size: Optional[int] = None,
    weather: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve relevant past assessments for RAG context.

    Uses hybrid approach:
    1. SQL filter by track, distance range, and optional criteria
    2. Rank by embedding similarity with bonus for matching attributes

    Args:
        db: Database session
        track: Track name (e.g., "Flemington")
        distance: Race distance in meters
        going: Track condition (e.g., "Good 3", "Heavy 8")
        race_class: Race class (e.g., "BM78", "Maiden")
        api_key: OpenAI API key for embeddings
        max_results: Maximum assessments to return
        age_restriction: Age restriction (e.g., "3YO", "Open")
        sex_restriction: Sex restriction (e.g., "F&M", "C&G")
        weight_type: Weight type (e.g., "Handicap", "WFA")
        field_size: Number of runners
        weather: Weather condition

    Returns:
        List of assessment dicts with key_learnings and context
    """
    # Derive state from track for filtering
    state = _get_state_from_track(track)

    # Step 1: SQL filter for candidates
    # Same track, distance within 200m
    query = (
        select(RaceAssessment)
        .where(
            and_(
                RaceAssessment.track == track,
                RaceAssessment.distance.between(distance - 200, distance + 200),
            )
        )
        .order_by(RaceAssessment.created_at.desc())
        .limit(30)  # Get more candidates for better ranking
    )
    result = await db.execute(query)
    candidates = list(result.scalars().all())

    # Broaden search if too few candidates - try same state first
    if len(candidates) < 5 and state:
        query = (
            select(RaceAssessment)
            .where(
                and_(
                    RaceAssessment.state == state,
                    RaceAssessment.distance.between(distance - 200, distance + 200),
                )
            )
            .order_by(RaceAssessment.created_at.desc())
            .limit(30)
        )
        result = await db.execute(query)
        state_candidates = list(result.scalars().all())
        # Add candidates we don't already have
        existing_ids = {c.id for c in candidates}
        for c in state_candidates:
            if c.id not in existing_ids:
                candidates.append(c)

    # Further broaden if still too few - any track at similar distance
    if len(candidates) < 5:
        query = (
            select(RaceAssessment)
            .where(
                RaceAssessment.distance.between(distance - 200, distance + 200)
            )
            .order_by(RaceAssessment.created_at.desc())
            .limit(30)
        )
        result = await db.execute(query)
        all_candidates = list(result.scalars().all())
        existing_ids = {c.id for c in candidates}
        for c in all_candidates:
            if c.id not in existing_ids:
                candidates.append(c)

    if not candidates:
        return []

    # Step 2: Score candidates using embedding similarity + attribute bonuses
    scored = []

    # Build rich query text for embedding
    query_parts = [track, f"{distance}m", race_class, going]
    if age_restriction:
        query_parts.append(age_restriction)
    if sex_restriction:
        query_parts.append(sex_restriction)
    if weight_type:
        query_parts.append(weight_type)
    if weather:
        query_parts.append(weather)
    if state:
        query_parts.append(state)
    query_text = " ".join(filter(None, query_parts))

    query_embedding = None
    if api_key:
        try:
            embedding_service = EmbeddingService(api_key=api_key)
            query_embedding = await embedding_service.get_embedding(query_text)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")

    for a in candidates:
        base_score = 0.5  # Default score

        # Embedding similarity (0-1 scale)
        if query_embedding and a.embedding:
            try:
                base_score = EmbeddingService.cosine_similarity(query_embedding, a.embedding)
            except Exception:
                pass

        # Attribute bonuses (add up to 0.3 total)
        bonus = 0.0

        # Same track is a strong signal
        if a.track == track:
            bonus += 0.1

        # Same state is useful
        if state and a.state == state:
            bonus += 0.03

        # Age restriction match
        if age_restriction and a.age_restriction == age_restriction:
            bonus += 0.05
        elif age_restriction and a.age_restriction and age_restriction.lower() in a.age_restriction.lower():
            bonus += 0.02

        # Sex restriction match
        if sex_restriction and a.sex_restriction == sex_restriction:
            bonus += 0.05

        # Weight type match
        if weight_type and a.weight_type == weight_type:
            bonus += 0.03

        # Similar field size (within 3 runners)
        if field_size and a.field_size:
            if abs(field_size - a.field_size) <= 3:
                bonus += 0.02

        # Similar going (wet/dry category match)
        going_lower = going.lower() if going else ""
        a_going_lower = (a.going or "").lower()
        is_wet_query = any(w in going_lower for w in ("heavy", "soft", "slow"))
        is_wet_stored = any(w in a_going_lower for w in ("heavy", "soft", "slow"))
        if is_wet_query == is_wet_stored:
            bonus += 0.02

        final_score = min(base_score + bonus, 1.0)
        scored.append((final_score, a))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [a for _, a in scored[:max_results]]

    # Format for context with all available fields
    return [a.to_retrieval_dict() | {"assessment": a.assessment} for a in candidates]


def build_rag_context_from_assessments(assessments: list[dict[str, Any]]) -> str:
    """Build a formatted RAG context string from retrieved assessments.

    This can be inserted into Early Mail prompts to provide learning context.
    """
    if not assessments:
        return ""

    lines = [
        "## Past Learnings from Similar Races",
        "The following insights were gathered from post-race reviews of similar races.",
        "Use these to inform your analysis, but each race is unique.",
        ""
    ]

    for a in assessments:
        hit_marker = "✓" if a.get("top_pick_hit") else "✗"
        pnl = a.get("total_pnl", 0)
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"${pnl:.2f}"

        # Build descriptive header with available context
        header_parts = [a.get("track", "Unknown"), f"{a.get('distance', 0)}m"]
        if a.get("class"):
            header_parts.append(a["class"])
        if a.get("going"):
            header_parts.append(a["going"])

        # Add race type details if available
        details = []
        if a.get("age_restriction"):
            details.append(a["age_restriction"])
        if a.get("sex_restriction"):
            details.append(a["sex_restriction"])
        if a.get("weight_type"):
            details.append(a["weight_type"])
        if a.get("field_size"):
            details.append(f"{a['field_size']} runners")
        if a.get("weather"):
            details.append(a["weather"])

        header = " ".join(header_parts)
        if details:
            header += f" ({', '.join(details)})"

        lines.append(f"**{header}** {hit_marker} {pnl_str}")
        lines.append(f"  → {a.get('key_learnings', 'No learnings recorded')}")
        lines.append("")

    return "\n".join(lines)
