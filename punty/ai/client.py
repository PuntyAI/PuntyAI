"""OpenAI API client wrapper."""

import logging
from typing import Optional

from openai import AsyncOpenAI

from punty.config import settings

logger = logging.getLogger(__name__)


class AIClient:
    """Wrapper for OpenAI API client."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 8000,
    ) -> str:
        """Generate content using OpenAI API.

        Args:
            system_prompt: System message defining Punty's personality
            user_prompt: The specific content request
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length

        Returns:
            Generated content as string
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.info(f"Generated {len(content)} characters with {self.model}")
            return content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_with_context(
        self,
        system_prompt: str,
        context: str,
        instruction: str,
        temperature: float = 0.8,
        max_tokens: int = 8000,
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
