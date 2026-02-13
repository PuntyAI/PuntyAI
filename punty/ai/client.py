"""OpenAI API client wrapper."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal

from openai import AsyncOpenAI, RateLimitError

from punty.config import settings

logger = logging.getLogger(__name__)

# Retry settings for rate limits
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 45  # seconds if we can't parse the wait time

# Reasoning effort levels for GPT-5.2 Responses API
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]

# Cost per million tokens (GPT-5.2 estimates — update as pricing changes)
TOKEN_COSTS = {
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
DEFAULT_COST = {"input": 5.00, "output": 15.00}


@dataclass
class TokenUsage:
    """Token usage from a single API call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AIClient:
    """Wrapper for OpenAI API client using Responses API for GPT-5.2."""

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
        reasoning_effort: ReasoningEffort = "high",
    ):
        self.model = model
        self._api_key = api_key
        self._client: Optional[AsyncOpenAI] = None
        self.reasoning_effort = reasoning_effort
        self.last_usage: Optional[TokenUsage] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            key = self._api_key or settings.openai_api_key
            if not key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client to free connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _parse_retry_after(self, error_message: str) -> float:
        """Extract retry delay from rate limit error message."""
        # Look for "Please try again in X.XXs" or "Please try again in Xs"
        match = re.search(r"try again in (\d+\.?\d*)s", str(error_message))
        if match:
            return float(match.group(1)) + 1  # Add 1s buffer
        return DEFAULT_RETRY_DELAY

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 32000,
    ) -> str:
        """Generate content using OpenAI Responses API with reasoning.

        Args:
            system_prompt: System message defining Punty's personality
            user_prompt: The specific content request
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length

        Returns:
            Generated content as string

        Raises:
            Exception: If all retries exhausted or non-rate-limit error
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                # Use Responses API for GPT-5.2 with reasoning
                if self.model.startswith("gpt-5"):
                    logger.info(f"Using {self.model} with reasoning_effort={self.reasoning_effort}")
                    response = await self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        reasoning={"effort": self.reasoning_effort},
                        max_output_tokens=max_tokens,
                    )
                    try:
                        content = response.output_text
                    except (AttributeError, TypeError, KeyError) as e:
                        logger.error(f"Malformed API response from {self.model}: {e}")
                        raise Exception(f"Malformed API response: {e}")
                    self._record_usage(response, is_responses_api=True)
                else:
                    # Fallback to Chat Completions for older models
                    logger.info(f"Using {self.model} (Chat Completions)")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    try:
                        content = response.choices[0].message.content
                    except (AttributeError, TypeError, IndexError, KeyError) as e:
                        logger.error(f"Malformed API response from {self.model}: {e}")
                        raise Exception(f"Malformed API response: {e}")
                    self._record_usage(response, is_responses_api=False)

                usage = self.last_usage
                usage_str = ""
                if usage:
                    usage_str = (
                        f" | tokens: {usage.input_tokens:,}in + {usage.output_tokens:,}out"
                        f" = {usage.total_tokens:,} | ${usage.estimated_cost:.3f}"
                    )
                logger.info(f"Generated {len(content)} chars with {self.model}{usage_str}")
                return content

            except RateLimitError as e:
                last_error = e
                retry_after = self._parse_retry_after(str(e))

                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES + 1}). "
                        f"Waiting {retry_after:.1f}s before retry..."
                    )
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(
                        f"Rate limit: All {MAX_RETRIES + 1} attempts exhausted. "
                        f"OpenAI TPM limit reached. Try again in ~1 minute."
                    )
                    raise

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise

        # Should not reach here, but just in case
        raise last_error

    def _record_usage(self, response, is_responses_api: bool) -> None:
        """Extract and store token usage from API response."""
        try:
            usage = TokenUsage(model=self.model)

            if is_responses_api and hasattr(response, "usage") and response.usage:
                usage.input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                usage.output_tokens = getattr(response.usage, "output_tokens", 0) or 0
                # Reasoning tokens are part of output_tokens for Responses API
                if hasattr(response.usage, "output_tokens_details"):
                    details = response.usage.output_tokens_details
                    usage.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            elif not is_responses_api and hasattr(response, "usage") and response.usage:
                usage.input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                usage.output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

            usage.total_tokens = usage.input_tokens + usage.output_tokens

            # Estimate cost
            costs = TOKEN_COSTS.get(self.model, DEFAULT_COST)
            usage.estimated_cost = (
                (usage.input_tokens / 1_000_000) * costs["input"]
                + (usage.output_tokens / 1_000_000) * costs["output"]
            )

            self.last_usage = usage
        except Exception as e:
            logger.debug(f"Could not extract token usage: {e}")
            self.last_usage = None

    async def generate_with_context(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.8,
        max_tokens: int = 32000,
    ) -> str:
        """Generate content with separate context and instruction.

        Args:
            system_prompt: System message for personality
            context: Racing data context
            instruction: What to generate

        Returns:
            Generated content
        """
        user_prompt = f"""## Racing Data Context

{context}

## Your Task

{instruction}"""

        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def review_and_fix(
        self,
        original_content: str,
        issue_type: str,
        notes: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Review and fix content based on feedback.

        Args:
            original_content: The content to fix
            issue_type: Type of issue (tone_wrong, too_long, etc.)
            notes: Additional feedback
            system_prompt: Optional personality prompt

        Returns:
            Fixed content
        """
        issue_instructions = {
            "tone_wrong": "Adjust the tone to be more aligned with Punty's cheeky, fun personality. Keep the facts but make it more entertaining.",
            "too_long": "Shorten this content significantly while keeping the key information. Be more concise.",
            "too_short": "Expand this content with more detail and personality. Add more racing insights.",
            "factually_incorrect": "Review and correct any factual errors. Ensure all horse names, odds, and race details are accurate.",
            "missing_info": "Add more relevant information that punters would want to know.",
            "not_funny": "Make this funnier! Add more Aussie humor, racing puns, and Punty's signature wit.",
        }

        instruction = issue_instructions.get(issue_type, "Improve this content based on feedback.")

        if notes:
            instruction += f"\n\nAdditional feedback: {notes}"

        user_prompt = f"""## Original Content

{original_content}

## Required Fix

{instruction}

Generate the improved version maintaining Punty's voice."""

        default_system = "You are helping Punty, a cheeky Australian horse racing content creator, improve their content."

        return await self.generate(
            system_prompt=system_prompt or default_system,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=2000,
        )


async def record_token_usage(
    db,
    usage: TokenUsage,
    content_type: str = "",
    meeting_id: str = "",
) -> None:
    """Record token usage to the database for cost tracking.

    Args:
        db: AsyncSession
        usage: TokenUsage from AIClient.last_usage
        content_type: e.g. 'early_mail', 'wrapup', 'assessment', 'fix'
        meeting_id: associated meeting ID if applicable
    """
    try:
        _text = __import__("sqlalchemy").text
        await db.execute(
            _text(
                "INSERT INTO token_usage "
                "(model, content_type, meeting_id, input_tokens, output_tokens, "
                "reasoning_tokens, total_tokens, estimated_cost, created_at) "
                "VALUES (:model, :content_type, :meeting_id, :input_tokens, "
                ":output_tokens, :reasoning_tokens, :total_tokens, "
                ":estimated_cost, :created_at)"
            ),
            {
                "model": usage.model,
                "content_type": content_type or "",
                "meeting_id": meeting_id or "",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "estimated_cost": usage.estimated_cost,
                "created_at": usage.timestamp.isoformat(),
            },
        )
        await db.commit()
    except Exception as e:
        logger.debug(f"Failed to record token usage: {e}")
