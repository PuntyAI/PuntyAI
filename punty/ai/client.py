"""OpenAI API client wrapper."""

import asyncio
import logging
import re
from typing import Optional, Literal

from openai import AsyncOpenAI, RateLimitError

from punty.config import settings

logger = logging.getLogger(__name__)

# Retry settings for rate limits
MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 45  # seconds if we can't parse the wait time

# Reasoning effort levels for GPT-5.2
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class AIClient:
    """Wrapper for OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: Optional[str] = None,
        reasoning_effort: ReasoningEffort = "medium",
    ):
        self.model = model
        self._api_key = api_key
        self._client: Optional[AsyncOpenAI] = None
        self.reasoning_effort = reasoning_effort

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            key = self._api_key or settings.openai_api_key
            if not key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=key)
        return self._client

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
        max_tokens: int = 16000,
    ) -> str:
        """Generate content using OpenAI API.

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
                # Build request parameters
                params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add reasoning for GPT-5+ models
                if self.model.startswith("gpt-5"):
                    params["reasoning"] = {"effort": self.reasoning_effort}
                    logger.info(f"Using {self.model} with reasoning_effort={self.reasoning_effort}")

                response = await self.client.chat.completions.create(**params)

                content = response.choices[0].message.content
                logger.info(f"Generated {len(content)} characters with {self.model}")
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

    async def generate_with_context(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.8,
        max_tokens: int = 16000,
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
