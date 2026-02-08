"""Content generation orchestrator."""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from openai import RateLimitError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from punty.ai.client import AIClient
from punty.context.builder import ContextBuilder
from punty.context.versioning import create_context_snapshot
from punty.models.content import Content, ContentStatus, ContentType

logger = logging.getLogger(__name__)

# Rate limit retry settings
RATE_LIMIT_PAUSE = 60  # seconds to pause when rate limited
MAX_RATE_LIMIT_RETRIES = 2  # retries at the stream level (on top of client retries)

# Load prompts from files
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load prompt template from file (or DB cache for personality)."""
    if name == "personality":
        cached = _personality_cache.get()
        if cached is not None:
            return cached
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return ""


class _PersonalityCache:
    """In-memory cache for personality prompt loaded from DB."""

    def __init__(self):
        self._content: str | None = None

    def get(self) -> str | None:
        return self._content

    def set(self, content: str):
        self._content = content


_personality_cache = _PersonalityCache()


class ContentGenerator:
    """Orchestrates content generation with context and prompts."""

    def __init__(self, db: AsyncSession, model: str = "gpt-5.2"):
        self.db = db
        self.ai_client = AIClient(model=model)
        self.context_builder = ContextBuilder(db)
        self._openai_key_loaded = False

    async def _ensure_openai_key(self):
        """Load OpenAI key from DB if not already set."""
        if self._openai_key_loaded:
            return
        from punty.models.settings import get_api_key
        from punty.config import settings
        db_key = await get_api_key(self.db, "openai_api_key", "")
        if db_key:
            self.ai_client._api_key = db_key
            self.ai_client._client = None  # Reset to pick up new key
        self._openai_key_loaded = True

    async def get_analysis_weights(self) -> str:
        """Load analysis weights from settings and format for prompt."""
        from punty.models.settings import AnalysisWeights

        result = await self.db.execute(
            select(AnalysisWeights).where(AnalysisWeights.id == "default")
        )
        weights = result.scalar_one_or_none()

        if weights:
            return weights.format_for_prompt()

        # Return default weights formatted
        return AnalysisWeights().format_for_prompt()

    async def get_setting(self, key: str, default: str = "") -> str:
        """Get a setting value from the database."""
        from punty.models.settings import AppSettings

        result = await self.db.execute(
            select(AppSettings).where(AppSettings.key == key)
        )
        setting = result.scalar_one_or_none()

        if setting:
            return setting.value

        # Check defaults
        if key in AppSettings.DEFAULTS:
            return AppSettings.DEFAULTS[key]["value"]

        return default

    async def generate_early_mail(
        self,
        meeting_id: str,
        save: bool = True,
        widen_if_favorites: bool = True,
    ) -> dict[str, Any]:
        """Generate Early Mail content for a meeting."""
        result = {}
        async for event in self.generate_early_mail_stream(meeting_id, save, widen_if_favorites):
            if event.get("status") == "complete":
                result = event.get("result", {})
            elif event.get("status") == "error":
                raise ValueError(event.get("label", "Generation failed"))
        return result

    async def generate_early_mail_stream(
        self,
        meeting_id: str,
        save: bool = True,
        widen_if_favorites: bool = True,
    ):
        """Generate Early Mail with SSE progress events."""
        from punty.scheduler.activity_log import log_generate_start, log_generate_complete, log_generate_error

        total_steps = 7
        step = 0
        venue_name = meeting_id  # Will be updated once we have context

        def evt(label, status="running", **extra):
            nonlocal step
            if status in ("running",):
                step += 1
            return {"step": step, "total": total_steps, "label": label, "status": status, **extra}

        try:
            await self._ensure_openai_key()
            yield evt("Building meeting context...")

            context = await self.context_builder.build_meeting_context(meeting_id)
            if not context:
                raise ValueError(f"Meeting not found: {meeting_id}")
            venue = context['meeting']['venue']
            venue_name = venue  # For activity log
            log_generate_start(venue)
            race_count = len(context.get('races', []))
            yield evt(f"Context built — {venue}, {race_count} races", "done")

            yield evt("Creating context snapshot...")
            snapshot = await create_context_snapshot(self.db, meeting_id)
            yield evt("Context snapshot saved", "done")

            yield evt("Loading prompts & analysis weights...")
            personality = load_prompt("personality")
            early_mail_prompt = load_prompt("early_mail")
            analysis_weights = await self.get_analysis_weights()
            yield evt("Prompts & weights loaded", "done")

            yield evt("Generating Early Mail with AI (this may take a moment)...")
            context_str = self._format_context_for_prompt(context)

            # Add learning from past predictions if available
            learning_context = await self._build_learning_context(context)
            if learning_context:
                context_str += "\n" + learning_context

            system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}
"""
            # Generate with rate limit retry, timeout, and status feedback
            raw_content = None
            for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    raw_content = await asyncio.wait_for(
                        self.ai_client.generate_with_context(
                            system_prompt=system_prompt,
                            context=context_str,
                            instruction=early_mail_prompt + f"\n\nGenerate Early Mail for {venue} on {context['meeting']['date']}",
                            temperature=0.8,
                        ),
                        timeout=300.0,  # 5 minute timeout (reasoning models need more time)
                    )
                    break  # Success
                except asyncio.TimeoutError:
                    logger.error(f"AI generation timed out for {venue} (attempt {attempt + 1})")
                    if attempt < MAX_RATE_LIMIT_RETRIES:
                        yield evt(f"Generation timed out — retrying (attempt {attempt + 2})...", "warning")
                    else:
                        raise Exception(f"AI generation timed out after {MAX_RATE_LIMIT_RETRIES + 1} attempts")
                except RateLimitError as e:
                    if attempt < MAX_RATE_LIMIT_RETRIES:
                        logger.warning(f"Rate limit hit for {venue}, pausing {RATE_LIMIT_PAUSE}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES + 1})")
                        yield evt(f"Rate limit reached — pausing for {RATE_LIMIT_PAUSE}s before retry...", "warning")
                        await asyncio.sleep(RATE_LIMIT_PAUSE)
                        yield evt(f"Retrying AI generation (attempt {attempt + 2})...")
                    else:
                        logger.error(f"Rate limit: All retries exhausted for {venue}")
                        raise  # Re-raise to be caught by outer exception handler

            if raw_content is None:
                raise Exception("AI generation failed - no content returned")

            yield evt("AI content generated", "done")

            favorites_detected = []
            yield evt("Checking for market favorites...")
            if widen_if_favorites:
                auto_widen = await self.get_setting("auto_widen_favorites", "false")
                if auto_widen.lower() == "true":
                    favorites_detected = await self._detect_favorites(raw_content, context)
                    max_favorites = int(await self.get_setting("max_favorites_per_card", "3"))

                    if len(favorites_detected) > max_favorites:
                        yield evt(f"Detected {len(favorites_detected)} favorites — widening selections...", "done")
                        step += 1
                        total_steps += 1
                        yield evt("Requesting wider selections from AI...")
                        raw_content = await self._request_wider_selections(
                            raw_content, favorites_detected, context, system_prompt
                        )
                        yield evt("Selections widened", "done")
                    else:
                        yield evt(f"Favorites check passed ({len(favorites_detected)} found)", "done")
                else:
                    yield evt("Auto-widen disabled, skipping", "done")
            else:
                yield evt("Favorites check skipped", "done")

            result = {
                "raw_content": raw_content,
                "meeting_id": meeting_id,
                "content_type": ContentType.EARLY_MAIL.value,
                "context_snapshot_id": snapshot["id"] if snapshot else None,
                "favorites_detected": favorites_detected,
            }

            if save:
                yield evt("Saving & formatting content...")
                content = await self._save_content(result, requires_review=True)
                result["content_id"] = content.id
                result["status"] = content.status
                yield evt("Content saved", "done")
            else:
                step += 1
                yield evt("Save skipped", "done")

            # Use "generation_done" instead of "complete" to avoid bulk generate JS thinking entire operation is done
            log_generate_complete(venue)
            yield {"step": total_steps, "total": total_steps, "label": f"Early Mail generated for {venue}", "status": "generation_done", "result": result}

        except Exception as e:
            logger.error(f"Early mail generation failed: {e}")
            log_generate_error(venue_name, str(e))
            yield {"step": step, "total": total_steps, "label": f"Error: {e}", "status": "error"}

    async def generate_initialise(
        self,
        meeting_id: str,
        save: bool = True,
    ) -> dict[str, Any]:
        """Generate meeting initialisation content."""
        await self._ensure_openai_key()
        logger.info(f"Generating Initialise for {meeting_id}")

        # Build context
        context = await self.context_builder.build_meeting_context(meeting_id)
        if not context:
            raise ValueError(f"Meeting not found: {meeting_id}")

        # Load prompts
        personality = load_prompt("personality")
        initialise_prompt = load_prompt("initialise")

        # Get analysis weights
        analysis_weights = await self.get_analysis_weights()
        initialise_prompt = initialise_prompt.replace("{ANALYSIS_WEIGHTS}", analysis_weights)

        # Format context
        context_str = self._format_context_for_prompt(context)

        # Generate
        raw_content = await self.ai_client.generate_with_context(
            system_prompt=personality,
            context=context_str,
            instruction=initialise_prompt + f"\n\nInitialise meeting for {context['meeting']['venue']} on {context['meeting']['date']}",
            temperature=0.7,
        )

        result = {
            "raw_content": raw_content,
            "meeting_id": meeting_id,
            "content_type": "initialise",
        }

        if save:
            content = await self._save_content(result, requires_review=True)
            result["content_id"] = content.id
            result["status"] = content.status

        return result

    async def request_widen_selections(
        self,
        content_id: str,
        feedback: Optional[str] = None,
    ) -> dict[str, Any]:
        """Request wider selections for existing content with too many favorites."""
        logger.info(f"Requesting wider selections for content {content_id}")

        # Get existing content
        result = await self.db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = result.scalar_one_or_none()
        if not content:
            raise ValueError(f"Content not found: {content_id}")

        # Build context
        context = await self.context_builder.build_meeting_context(content.meeting_id)

        # Detect favorites in current content
        favorites_detected = await self._detect_favorites(content.raw_content, context)

        # Load prompts
        personality = load_prompt("personality")
        widen_prompt = load_prompt("widen_selections")

        # Get analysis weights
        analysis_weights = await self.get_analysis_weights()

        # Format widen prompt with detected favorites
        favorites_threshold = float(await self.get_setting("favorites_threshold", "2.50"))
        replace_count = max(1, len(favorites_detected) - 2)  # Keep at most 2 favorites

        widen_prompt = widen_prompt.replace("{FAVORITE_COUNT}", str(len(favorites_detected)))
        widen_prompt = widen_prompt.replace("{FAVORITES_THRESHOLD}", f"{favorites_threshold:.2f}")
        widen_prompt = widen_prompt.replace("{REPLACE_COUNT}", str(replace_count))

        if feedback:
            widen_prompt += f"\n\nAdditional feedback from user:\n{feedback}"

        # Format context
        context_str = self._format_context_for_prompt(context)

        system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}

## Current Content Being Revised
{content.raw_content}

## Favorites Detected
{json.dumps(favorites_detected, indent=2)}
"""

        # Generate revised content
        raw_content = await self.ai_client.generate_with_context(
            system_prompt=system_prompt,
            context=context_str,
            instruction=widen_prompt,
            temperature=0.8,
        )

        # Save as new version
        from punty.formatters.twitter import format_twitter

        content.raw_content = raw_content
        content.twitter_formatted = format_twitter(raw_content, content.content_type)
        content.review_notes = f"Widened selections (was {len(favorites_detected)} favorites)"

        await self.db.commit()

        return {
            "content_id": content.id,
            "raw_content": raw_content,
            "favorites_before": len(favorites_detected),
            "status": "widened",
        }

    async def _build_learning_context(self, context: dict) -> str:
        """Build learning context from past predictions for inclusion in prompt.

        Finds similar past situations and includes insights about what worked/didn't.
        Uses both individual race_memories AND structured race_assessments.
        """
        try:
            from punty.memory.store import MemoryStore
            from punty.memory.embeddings import EmbeddingService
            from punty.memory.assessment import retrieve_assessment_context, build_rag_context_from_assessments
            from punty.models.settings import get_api_key

            memory_store = MemoryStore(self.db, EmbeddingService())
            stats = await memory_store.get_stats()

            learning_parts = []
            meeting = context.get("meeting", {})
            track = meeting.get("venue", "")
            going = meeting.get("track_condition", "Unknown")

            # Get API key for embedding-based retrieval
            api_key = await get_api_key(self.db, "openai_api_key")

            # First, retrieve track-level learnings from race assessments
            races = context.get("races", [])
            assessment_learnings = []
            seen_assessments = set()
            weather = meeting.get("weather_condition") or meeting.get("weather")

            for race in races:
                distance = race.get("distance") or 1200
                race_class = race.get("class") or "Unknown"
                age_restriction = race.get("age_restriction")
                weight_type = race.get("weight_type")
                field_size = race.get("field_size")

                # Derive sex restriction from race name
                race_name = race.get("name", "")
                sex_restriction = None
                race_name_lower = race_name.lower()
                if "fillies" in race_name_lower or "mares" in race_name_lower:
                    sex_restriction = "F&M"
                elif "colts" in race_name_lower or "geldings" in race_name_lower:
                    sex_restriction = "C&G"

                assessments = await retrieve_assessment_context(
                    self.db, track, distance, going, race_class,
                    api_key=api_key, max_results=2,
                    age_restriction=age_restriction,
                    sex_restriction=sex_restriction,
                    weight_type=weight_type,
                    field_size=field_size,
                    weather=weather,
                )

                for a in assessments:
                    # Deduplicate by key learnings
                    learning_key = a.get("key_learnings", "")
                    if learning_key and learning_key not in seen_assessments:
                        seen_assessments.add(learning_key)
                        assessment_learnings.append(a)

            # Add assessment-based learnings
            if assessment_learnings:
                rag_context = build_rag_context_from_assessments(assessment_learnings[:5])
                if rag_context:
                    learning_parts.append(rag_context)
                    learning_parts.append("")

            # Only include runner-level memories if we have enough settled data
            if stats.get("settled_memories", 0) >= 10:
                runner_learning_parts = [
                    "## INDIVIDUAL RUNNER PATTERNS",
                    f"(Based on {stats['settled_memories']} settled predictions, {stats['hit_rate']:.1f}% hit rate, avg PNL: {stats['avg_pnl']:+.2f}U)",
                    "",
                ]

                # For each race, find similar past situations for runners
                has_runner_content = False
                for race in races:
                    race_context = {
                        "track_condition": going,
                        "distance": race.get("distance"),
                        "class": race.get("class"),
                    }

                    runners = race.get("runners", [])[:3]  # Top 3 by odds
                    for runner in runners:
                        if runner.get("scratched"):
                            continue

                        runner_data = {
                            "horse_name": runner.get("horse_name"),
                            "form": runner.get("form"),
                            "current_odds": runner.get("current_odds"),
                            "speed_map_position": runner.get("speed_map_position"),
                            "odds_movement": runner.get("odds_movement"),
                            "pf_map_factor": runner.get("pf_map_factor"),
                        }

                        learning = await memory_store.build_learning_context(
                            race_context, runner_data, max_memories=2
                        )

                        if learning:
                            runner_learning_parts.append(f"**R{race['race_number']} {runner.get('horse_name')}:**")
                            runner_learning_parts.append(learning)
                            runner_learning_parts.append("")
                            has_runner_content = True

                if has_runner_content:
                    learning_parts.extend(runner_learning_parts)

            if learning_parts:
                return "\n".join(learning_parts)
            return ""

        except Exception as e:
            logger.warning(f"Failed to build learning context: {e}")
            return ""

    async def _detect_favorites(
        self,
        content: str,
        context: dict,
    ) -> list[dict]:
        """Detect selections that are market favorites based on odds threshold."""
        favorites_threshold = float(await self.get_setting("favorites_threshold", "2.50"))
        favorites = []

        # Build a map of horse names to their odds from context
        horse_odds = {}
        for race in context.get("races", []):
            for runner in race.get("runners", []):
                horse_name = runner.get("horse_name", "").upper()
                odds = runner.get("current_odds")
                if odds and horse_name:
                    horse_odds[horse_name] = float(odds)

        # Look for horse names mentioned in content and check their odds
        # This is a simple heuristic - look for capitalized horse names
        for horse_name, odds in horse_odds.items():
            if odds <= favorites_threshold:
                # Check if this horse appears to be a selection (near key phrases)
                patterns = [
                    rf'\*{re.escape(horse_name)}\*',
                    rf'(?:BEST BET|NEXT BEST|TOP|Big 3).*{re.escape(horse_name)}',
                    rf'{re.escape(horse_name)}.*\$\d+\.\d+',
                ]
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        favorites.append({
                            "horse": horse_name,
                            "odds": odds,
                            "threshold": favorites_threshold,
                        })
                        break

        return favorites

    async def _request_wider_selections(
        self,
        current_content: str,
        favorites_detected: list[dict],
        context: dict,
        system_prompt: str,
    ) -> str:
        """Ask AI to reconsider and provide wider selections."""
        widen_prompt = load_prompt("widen_selections")

        favorites_threshold = float(await self.get_setting("favorites_threshold", "2.50"))
        replace_count = max(1, len(favorites_detected) - 2)

        widen_prompt = widen_prompt.replace("{FAVORITE_COUNT}", str(len(favorites_detected)))
        widen_prompt = widen_prompt.replace("{FAVORITES_THRESHOLD}", f"{favorites_threshold:.2f}")
        widen_prompt = widen_prompt.replace("{REPLACE_COUNT}", str(replace_count))

        context_str = self._format_context_for_prompt(context)

        followup_prompt = f"""## Your Previous Output
{current_content}

## Favorites Detected
{json.dumps(favorites_detected, indent=2)}

{widen_prompt}

Please provide a COMPLETE revised Early Mail with wider selections. Output the full content, not just the changes.
"""

        revised_content = await self.ai_client.generate_with_context(
            system_prompt=system_prompt,
            context=context_str,
            instruction=followup_prompt,
            temperature=0.85,
        )

        return revised_content

    async def generate_race_preview(
        self,
        meeting_id: str,
        race_number: int,
        save: bool = True,
    ) -> dict[str, Any]:
        """Generate preview for a specific race."""
        await self._ensure_openai_key()
        logger.info(f"Generating race preview for R{race_number} at {meeting_id}")

        # Build race-specific context
        context = await self.context_builder.build_race_context(meeting_id, race_number)
        if not context:
            raise ValueError(f"Race not found: R{race_number} at {meeting_id}")

        # Load prompts
        personality = load_prompt("personality")
        preview_prompt = load_prompt("race_preview")
        analysis_weights = await self.get_analysis_weights()

        # Format context
        context_str = json.dumps(context, indent=2, default=str)

        system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}
"""

        # Generate
        raw_content = await self.ai_client.generate_with_context(
            system_prompt=system_prompt,
            context=context_str,
            instruction=preview_prompt + f"\n\nGenerate preview for Race {race_number}",
            temperature=0.8,
        )

        result = {
            "raw_content": raw_content,
            "meeting_id": meeting_id,
            "race_number": race_number,
            "content_type": ContentType.RACE_PREVIEW.value,
        }

        if save:
            content = await self._save_content(result, requires_review=False)
            result["content_id"] = content.id
            result["status"] = content.status

        return result

    async def generate_results(
        self,
        meeting_id: str,
        race_number: int,
        save: bool = True,
    ) -> dict[str, Any]:
        """Generate results commentary for a race."""
        result = {}
        async for event in self.generate_results_stream(meeting_id, race_number, save):
            if event.get("status") == "complete":
                result = event.get("result", {})
            elif event.get("status") == "error":
                raise ValueError(event.get("label", "Generation failed"))
        return result

    async def generate_results_stream(
        self,
        meeting_id: str,
        race_number: int,
        save: bool = True,
    ):
        """Generate results commentary with SSE progress events."""
        total_steps = 5
        step = 0

        def evt(label, status="running", **extra):
            nonlocal step
            if status == "running":
                step += 1
            return {"step": step, "total": total_steps, "label": label, "status": status, **extra}

        try:
            await self._ensure_openai_key()
            yield evt("Building results context...")
            context = await self.context_builder.build_results_context(meeting_id, race_number)
            if not context:
                raise ValueError(f"Race not found: R{race_number} at {meeting_id}")
            if not context.get("race", {}).get("results"):
                raise ValueError(f"No results data for R{race_number} yet — run Check Results first")
            yield evt(f"Context built — Race {race_number}", "done")

            yield evt("Loading prompts...")
            personality = load_prompt("personality")
            results_prompt = load_prompt("results")
            analysis_weights = await self.get_analysis_weights()
            yield evt("Prompts loaded", "done")

            yield evt("Generating results commentary with AI...")
            context_str = json.dumps(context, indent=2, default=str)
            system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}
"""
            raw_content = await self.ai_client.generate_with_context(
                system_prompt=system_prompt,
                context=context_str,
                instruction=results_prompt + f"\n\nGenerate results commentary for Race {race_number}",
                temperature=0.8,
            )
            yield evt("AI content generated", "done")

            result = {
                "raw_content": raw_content,
                "meeting_id": meeting_id,
                "race_number": race_number,
                "content_type": ContentType.RESULTS.value,
            }

            if save:
                yield evt("Saving content...")
                content = await self._save_content(result, requires_review=True)
                result["content_id"] = content.id
                result["status"] = content.status
                yield evt("Content saved", "done")
            else:
                step += 1
                yield evt("Save skipped", "done")

            yield {"step": total_steps, "total": total_steps, "label": f"Results generated for Race {race_number}", "status": "generation_done", "result": result}

        except Exception as e:
            logger.error(f"Results generation failed: {e}")
            yield {"step": step, "total": total_steps, "label": f"Error: {e}", "status": "error"}

    async def generate_meeting_wrapup_stream(
        self,
        meeting_id: str,
        save: bool = True,
    ):
        """Generate meeting punt review with SSE progress events."""
        total_steps = 5
        step = 0

        def evt(label, status="running", **extra):
            nonlocal step
            if status == "running":
                step += 1
            return {"step": step, "total": total_steps, "label": label, "status": status, **extra}

        from punty.scheduler.activity_log import log_generate_start, log_generate_complete, log_generate_error
        venue_name = "Unknown"

        try:
            await self._ensure_openai_key()
            yield evt("Building meeting wrap-up context...")
            context = await self.context_builder.build_wrapup_context(meeting_id)
            if not context:
                raise ValueError(f"Meeting not found: {meeting_id}")
            if not context.get("race_summaries"):
                raise ValueError("No completed races yet — check results first")
            venue = context["meeting"]["venue"]
            venue_name = venue
            log_generate_start(venue, content_type="Wrap-up")
            yield evt(f"Context built — {venue} ({len(context['race_summaries'])} races)", "done")

            yield evt("Loading prompts & weights...")
            personality = load_prompt("personality")
            wrapup_prompt = load_prompt("wrap_up")
            analysis_weights = await self.get_analysis_weights()
            yield evt("Prompts loaded", "done")

            yield evt("Generating punt review with AI...")
            context_str = json.dumps(context, indent=2, default=str)
            system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}
"""
            # Generate with rate limit retry and status feedback
            raw_content = None
            for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    raw_content = await self.ai_client.generate_with_context(
                        system_prompt=system_prompt,
                        context=context_str,
                        instruction=wrapup_prompt + f"\n\nGenerate meeting wrap-up for {venue} on {context['meeting']['date']}",
                        temperature=0.8,
                    )
                    break  # Success
                except RateLimitError as e:
                    if attempt < MAX_RATE_LIMIT_RETRIES:
                        logger.warning(f"Rate limit hit for wrapup {venue}, pausing {RATE_LIMIT_PAUSE}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES + 1})")
                        yield evt(f"Rate limit reached — pausing for {RATE_LIMIT_PAUSE}s before retry...", "warning")
                        await asyncio.sleep(RATE_LIMIT_PAUSE)
                        yield evt(f"Retrying AI generation (attempt {attempt + 2})...")
                    else:
                        logger.error(f"Rate limit: All retries exhausted for wrapup {venue}")
                        raise

            if raw_content is None:
                raise Exception("AI generation failed - no content returned")

            yield evt("AI content generated", "done")

            result = {
                "raw_content": raw_content,
                "meeting_id": meeting_id,
                "content_type": ContentType.MEETING_WRAPUP.value,
            }

            if save:
                yield evt("Saving content...")
                content = await self._save_content(result, requires_review=True)
                result["content_id"] = content.id
                result["status"] = content.status
                yield evt("Content saved", "done")
            else:
                step += 1
                yield evt("Save skipped", "done")

            log_generate_complete(venue, content_type="Wrap-up")
            yield {"step": total_steps, "total": total_steps, "label": f"Punt Review generated for {venue}", "status": "generation_done", "result": result}

        except Exception as e:
            logger.error(f"Meeting wrapup generation failed: {e}")
            log_generate_error(venue_name, str(e), content_type="Wrap-up")
            yield {"step": step, "total": total_steps, "label": f"Error: {e}", "status": "error"}

    async def generate_meeting_wrapup(
        self,
        meeting_id: str,
        save: bool = True,
    ) -> dict[str, Any]:
        """Generate meeting wrap-up content."""
        await self._ensure_openai_key()
        logger.info(f"Generating meeting wrap-up for {meeting_id}")

        context = await self.context_builder.build_wrapup_context(meeting_id)
        if not context:
            raise ValueError(f"Meeting not found: {meeting_id}")

        personality = load_prompt("personality")
        wrapup_prompt = load_prompt("wrap_up")
        analysis_weights = await self.get_analysis_weights()

        context_str = json.dumps(context, indent=2, default=str)
        system_prompt = f"""{personality}

## Analysis Framework Weights
{analysis_weights}
"""

        raw_content = await self.ai_client.generate_with_context(
            system_prompt=system_prompt,
            context=context_str,
            instruction=wrapup_prompt + f"\n\nGenerate meeting wrap-up for {context['meeting']['venue']} on {context['meeting']['date']}",
            temperature=0.8,
        )

        result = {
            "raw_content": raw_content,
            "meeting_id": meeting_id,
            "content_type": ContentType.MEETING_WRAPUP.value,
        }

        if save:
            content = await self._save_content(result, requires_review=True)
            result["content_id"] = content.id
            result["status"] = content.status

        return result

    async def generate_update_alert(
        self,
        meeting_id: str,
        changes: list[dict],
        save: bool = True,
    ) -> dict[str, Any]:
        """Generate alert when significant context changes detected."""
        await self._ensure_openai_key()
        logger.info(f"Generating update alert for {meeting_id}")

        # Build full context
        context = await self.context_builder.build_meeting_context(meeting_id)

        # Load prompts
        personality = load_prompt("personality")
        update_prompt = load_prompt("speed_map_update")

        # Format changes
        changes_str = "\n".join([f"- {c['description']}" for c in changes])

        # Generate
        raw_content = await self.ai_client.generate_with_context(
            system_prompt=personality,
            context=f"Changes detected:\n{changes_str}\n\nFull context:\n{json.dumps(context, indent=2, default=str)}",
            instruction=update_prompt,
            temperature=0.7,
        )

        result = {
            "raw_content": raw_content,
            "meeting_id": meeting_id,
            "content_type": ContentType.UPDATE_ALERT.value,
            "changes": changes,
        }

        if save:
            content = await self._save_content(result, requires_review=True)
            result["content_id"] = content.id
            result["status"] = content.status

        return result

    @staticmethod
    def _get_sequence_lanes(total_races: int) -> dict:
        """Get quaddie, early quaddie and big 6 race ranges based on meeting size."""
        rules = {
            7:  {"early_quad": (1, 4), "quaddie": (4, 7), "big6": None},
            8:  {"early_quad": (1, 4), "quaddie": (5, 8), "big6": (3, 8)},
            9:  {"early_quad": (2, 5), "quaddie": (6, 9), "big6": (4, 9)},
            10: {"early_quad": (3, 6), "quaddie": (7, 10), "big6": (5, 10)},
            11: {"early_quad": (4, 7), "quaddie": (8, 11), "big6": (6, 11)},
            12: {"early_quad": (5, 8), "quaddie": (9, 12), "big6": (7, 12)},
        }
        return rules.get(total_races, rules.get(min(rules.keys(), key=lambda k: abs(k - total_races)), {}))

    def _format_context_for_prompt(self, context: dict) -> str:
        """Format context dict into readable prompt text."""
        meeting = context.get("meeting", {})
        races = context.get("races", [])
        summary = context.get("summary", {})

        parts = [
            f"## {meeting.get('venue', 'Unknown')} - {meeting.get('date', 'Unknown')}",
            f"Track: {meeting.get('track_condition', 'TBC')}",
            f"Rail: {meeting.get('rail_position', 'TBC')}",
            f"Weather: {meeting.get('weather', 'TBC')}",
            "",
            f"**{summary.get('total_races', 0)} races, {summary.get('total_runners', 0)} runners, {summary.get('scratchings', 0)} scratchings**",
            "",
        ]

        for race in races:
            parts.append(f"### Race {race['race_number']}: {race['name']}")
            parts.append(f"{race['distance']}m | {race.get('class', 'Open')} | ${race.get('prize_money', 0):,}")

            # Pace scenario
            analysis = race.get("analysis", {})
            parts.append(f"Pace: {analysis.get('pace_scenario', 'unknown').replace('_', ' ').title()}")

            if analysis.get("likely_leaders"):
                parts.append(f"Leaders: {', '.join(analysis['likely_leaders'])}")

            # Show pace-advantaged/disadvantaged runners from PF data
            if analysis.get("pace_advantaged"):
                adv_list = [f"{p['horse']} ({p['map_factor']:.2f})" for p in analysis["pace_advantaged"]]
                parts.append(f"Pace Advantaged: {', '.join(adv_list)}")
            if analysis.get("pace_disadvantaged"):
                dis_list = [f"{p['horse']} ({p['map_factor']:.2f})" for p in analysis["pace_disadvantaged"]]
                parts.append(f"Pace Disadvantaged: {', '.join(dis_list)}")

            parts.append("")
            parts.append("| No. | Horse | Barrier | Jockey | Odds | Form | Speed Map | Market Move | PF Speed | PF Map |")
            parts.append("|-----|-------|---------|--------|------|------|-----------|-------------|----------|--------|")

            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                saddlecloth = runner.get("saddlecloth", runner.get('barrier', '-'))
                # Format PF fields
                pf_speed = runner.get("pf_speed_rank") or "-"
                pf_map = f"{runner['pf_map_factor']:.2f}" if runner.get("pf_map_factor") else "-"
                # Format market movement
                market_move = "-"
                mm = runner.get("market_movement")
                if mm and mm.get("direction") and mm.get("direction") != "stable":
                    from_price = mm.get("from")
                    to_price = mm.get("to")
                    direction = mm.get("direction")
                    if from_price and to_price:
                        if direction in ("heavy_support", "firming"):
                            market_move = f"${from_price:.0f}→${to_price:.0f} IN"
                        else:
                            market_move = f"${from_price:.0f}→${to_price:.0f} OUT"
                parts.append(
                    f"| {saddlecloth} | {runner.get('horse_name', 'Unknown')} | "
                    f"{runner.get('barrier', '-')} | {runner.get('jockey', '-')} | "
                    f"${runner.get('current_odds', '-')} | {runner.get('form', '-')} | "
                    f"{runner.get('speed_map_position', '-')} | {market_move} | {pf_speed} | {pf_map} |"
                )

            # Add detailed runner stats below the table
            parts.append("")
            parts.append("**Runner Details:**")
            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                details = []
                horse = runner.get("horse_name", "Unknown")
                saddlecloth = runner.get("saddlecloth", "?")

                # Days since last run
                days = runner.get("days_since_last_run")
                if days:
                    details.append(f"{days}d")

                # Career record
                if runner.get("career_record"):
                    details.append(f"Career: {runner['career_record']}")

                # Gear changes (important!)
                if runner.get("gear_changes"):
                    details.append(f"GEAR: {runner['gear_changes']}")
                elif runner.get("gear"):
                    details.append(f"Gear: {runner['gear']}")

                # First/second up
                if runner.get("first_up_stats"):
                    details.append(f"1st-up: {runner['first_up_stats']}")
                if runner.get("second_up_stats"):
                    details.append(f"2nd-up: {runner['second_up_stats']}")

                # Track/distance stats
                if runner.get("track_stats"):
                    details.append(f"Track: {runner['track_stats']}")
                if runner.get("distance_stats"):
                    details.append(f"Dist: {runner['distance_stats']}")

                # Track condition stats
                track_cond = meeting.get("track_condition", "").lower()
                if "heavy" in track_cond and runner.get("heavy_track_stats"):
                    details.append(f"Heavy: {runner['heavy_track_stats']}")
                elif "soft" in track_cond and runner.get("soft_track_stats"):
                    details.append(f"Soft: {runner['soft_track_stats']}")
                elif runner.get("good_track_stats"):
                    details.append(f"Good: {runner['good_track_stats']}")

                # Jockey/trainer at track
                if runner.get("jockey_stats"):
                    details.append(f"Jockey@track: {runner['jockey_stats']}")
                if runner.get("trainer_stats"):
                    details.append(f"Trainer@track: {runner['trainer_stats']}")

                # Class stats
                if runner.get("class_stats"):
                    details.append(f"Class: {runner['class_stats']}")

                # Sectionals (last run)
                sec_400 = runner.get("sectional_400")
                sec_800 = runner.get("sectional_800")
                if sec_400 or sec_800:
                    sec_str = f"L400: {sec_400}s" if sec_400 else ""
                    if sec_800:
                        sec_str += f" L800: {sec_800}s" if sec_str else f"L800: {sec_800}s"
                    details.append(sec_str)

                # Pedigree
                sire = runner.get("sire")
                dam = runner.get("dam")
                if sire:
                    pedigree = f"by {sire}"
                    if dam:
                        pedigree += f" x {dam}"
                    details.append(pedigree)

                if details:
                    parts.append(f"- No.{saddlecloth} {horse}: {' | '.join(details)}")

            # Tipster comments / analysis for this race
            race_comments = []
            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                comment = runner.get("comment_long") or runner.get("comment_short") or runner.get("comments")
                if comment and len(comment) > 10:
                    race_comments.append({
                        "horse": runner.get("horse_name"),
                        "comment": comment[:200],  # Truncate long comments
                    })
            if race_comments:
                parts.append("")
                parts.append("**Tipster/Form Comments:**")
                for c in race_comments[:6]:  # Top 6 comments per race
                    parts.append(f"- {c['horse']}: {c['comment']}")

            # Stewards comments
            stewards = []
            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                stew = runner.get("stewards_comment")
                if stew and len(stew) > 5:
                    stewards.append({
                        "horse": runner.get("horse_name"),
                        "comment": stew[:150],
                    })
            if stewards:
                parts.append("")
                parts.append("**Stewards Comments:**")
                for s in stewards:
                    parts.append(f"- {s['horse']}: {s['comment']}")

            # Extended form history (past starts with sectionals)
            form_histories = []
            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                fh_raw = runner.get("form_history")
                if fh_raw:
                    try:
                        fh = json.loads(fh_raw) if isinstance(fh_raw, str) else fh_raw
                        if fh and len(fh) > 0:
                            form_histories.append({
                                "horse": runner.get("horse_name"),
                                "saddlecloth": runner.get("saddlecloth"),
                                "starts": fh[:5],  # Last 5 starts
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass

            if form_histories:
                parts.append("")
                parts.append("**Extended Form (Last 5 Starts):**")
                for fh in form_histories[:8]:  # Show up to 8 horses with form
                    horse = fh["horse"]
                    starts_str = []
                    for start in fh["starts"]:
                        # Format: venue distance pos (margin) track
                        venue = start.get("venue", "?")
                        dist = start.get("distance", "?")
                        pos = start.get("pos", "?")
                        margin = start.get("margin", "")
                        track = start.get("track", "")
                        settled = start.get("settled", "")
                        at400 = start.get("at400", "")

                        start_info = f"{venue} {dist}m-{pos}"
                        if margin:
                            start_info += f"({margin})"
                        if track:
                            start_info += f" {track}"
                        if settled:
                            start_info += f" Sett:{settled}"
                        if at400:
                            start_info += f" 400:{at400}"
                        starts_str.append(start_info)

                    parts.append(f"- No.{fh['saddlecloth']} {horse}: {' | '.join(starts_str)}")

            parts.append("")

        # Add summary insights
        if summary.get("favorites"):
            parts.append("## Market Favorites")
            for fav in summary["favorites"]:
                parts.append(f"- Race {fav['race']}: {fav['horse']} @ ${fav['odds']}")

        if summary.get("roughies"):
            parts.append("\n## Value/Roughies")
            for rough in summary["roughies"]:
                parts.append(f"- Race {rough['race']}: {rough['horse']} @ ${rough['odds']} (form: {rough['form']})")

        # Collect market movers from all races
        market_in = []
        market_out = []
        for race in races:
            race_num = race.get("race_number")
            analysis = race.get("analysis", {})
            for mover in analysis.get("market_movers", []):
                entry = {
                    "race": race_num,
                    "horse": mover.get("horse"),
                    "from": mover.get("from"),
                    "to": mover.get("to"),
                    "movement": mover.get("movement"),
                    "summary": mover.get("summary"),
                }
                if mover.get("direction") == "in":
                    market_in.append(entry)
                else:
                    market_out.append(entry)

        if market_in:
            parts.append("\n## MARKET SUPPORT (Horses Being Backed)")
            for m in market_in:
                label = "HEAVY" if m.get("movement") == "heavy_support" else "Firming"
                move_summary = m.get("summary") or f"${m.get('from', 0):.2f} → ${m.get('to', 0):.2f}"
                parts.append(f"- Race {m['race']}: {m['horse']} — {move_summary} [{label}]")

        if market_out:
            parts.append("\n## MARKET DRIFTERS (Horses Easing)")
            for m in market_out:
                label = "BIG DRIFT" if m.get("movement") == "big_drift" else "Drifting"
                move_summary = m.get("summary") or f"${m.get('from', 0):.2f} → ${m.get('to', 0):.2f}"
                parts.append(f"- Race {m['race']}: {m['horse']} — {move_summary} [{label}]")

        # Collect gear changes across all races
        gear_changes = []
        first_uppers = []
        for race in races:
            race_num = race.get("race_number")
            for runner in race.get("runners", []):
                if runner.get("scratched"):
                    continue
                horse = runner.get("horse_name")
                odds = runner.get("current_odds")
                # Gear changes
                if runner.get("gear_changes"):
                    gear_changes.append({
                        "race": race_num,
                        "horse": horse,
                        "change": runner["gear_changes"],
                        "odds": odds,
                    })
                # First up runners (check days_since_last_run or first_up_stats)
                days = runner.get("days_since_last_run")
                first_up = runner.get("first_up_stats")
                if days and days > 60:  # Resuming after 60+ days
                    first_uppers.append({
                        "race": race_num,
                        "horse": horse,
                        "days": days,
                        "stats": first_up,
                        "odds": odds,
                    })

        if gear_changes:
            parts.append("\n## GEAR CHANGES (Important!)")
            for g in gear_changes:
                parts.append(f"- Race {g['race']}: {g['horse']} — {g['change']} (${g['odds']})")

        if first_uppers:
            parts.append("\n## FIRST-UP RUNNERS (Resuming)")
            for f in first_uppers:
                stats_str = f" | 1st-up record: {f['stats']}" if f.get("stats") else ""
                parts.append(f"- Race {f['race']}: {f['horse']} — {f['days']} days off (${f['odds']}){stats_str}")

        # Sequence lanes (quaddie, early quaddie, big 6)
        total_races = summary.get("total_races", len(races))
        sequences = self._get_sequence_lanes(total_races)
        if sequences:
            parts.append("\n## SEQUENCE LANES (use these exact race ranges)")
            eq = sequences.get("early_quad")
            q = sequences.get("quaddie")
            b6 = sequences.get("big6")
            if eq:
                parts.append(f"- EARLY QUADDIE: Races {eq[0]}-{eq[1]}")
            if q:
                parts.append(f"- QUADDIE (main): Races {q[0]}-{q[1]}")
            if b6:
                parts.append(f"- BIG 6: Races {b6[0]}-{b6[1]}")
            elif total_races <= 7:
                parts.append("- BIG 6: Not applicable (7 race meeting)")

        return "\n".join(parts)

    async def _save_content(
        self,
        result: dict,
        requires_review: bool = True,
    ) -> Content:
        """Save generated content to database."""
        from punty.models.meeting import Race
        from punty.formatters.twitter import format_twitter

        # Get race_id if race_number provided
        race_id = None
        if result.get("race_number"):
            race_result = await self.db.execute(
                select(Race).where(
                    Race.meeting_id == result["meeting_id"],
                    Race.race_number == result["race_number"],
                )
            )
            race = race_result.scalar_one_or_none()
            if race:
                race_id = race.id

        # Format for platforms
        raw_content = result["raw_content"]
        content_type = result["content_type"]
        twitter_formatted = format_twitter(raw_content, content_type)

        content = Content(
            id=str(uuid.uuid4()),
            meeting_id=result["meeting_id"],
            race_id=race_id,
            content_type=result["content_type"],
            context_snapshot_id=result.get("context_snapshot_id"),
            status=ContentStatus.PENDING_REVIEW.value if requires_review else ContentStatus.APPROVED.value,
            requires_review=requires_review,
            raw_content=raw_content,
            twitter_formatted=twitter_formatted,
        )

        self.db.add(content)
        await self.db.commit()

        logger.info(f"Saved content {content.id} ({content.content_type}) - status: {content.status}")

        return content
