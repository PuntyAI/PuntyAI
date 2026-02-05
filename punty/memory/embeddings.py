"""Embedding service for generating and comparing race context embeddings."""

import json
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for race contexts and find similar situations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Lazy init OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                from punty.models.database import async_session
                from punty.models.settings import get_api_key

                # Get API key from DB if not provided
                if not self.api_key:
                    async with async_session() as db:
                        self.api_key = await get_api_key(db, "openai_api_key")

                if self.api_key:
                    self._client = AsyncOpenAI(api_key=self.api_key)
                else:
                    logger.warning("No OpenAI API key available for embeddings")
            except ImportError:
                logger.warning("openai package not installed")
        return self._client

    def build_context_text(self, context: dict[str, Any], runner: dict[str, Any]) -> str:
        """Build a text representation of race context for embedding.

        The text should capture the key factors that influence predictions:
        - Race conditions (track, distance, class, rail)
        - Runner profile (form, age, barrier, jockey, trainer)
        - Market signals (odds, movement)
        - Pace dynamics (speed map position, tempo)
        """
        parts = []

        # Race conditions
        if context.get("track_condition"):
            parts.append(f"Track: {context['track_condition']}")
        if context.get("distance"):
            parts.append(f"Distance: {context['distance']}m")
        if context.get("class"):
            parts.append(f"Class: {context['class']}")
        if context.get("rail_position"):
            parts.append(f"Rail: {context['rail_position']}")
        if context.get("tempo"):
            parts.append(f"Tempo: {context['tempo']}")

        # Runner profile
        if runner.get("form"):
            parts.append(f"Form: {runner['form']}")
        if runner.get("horse_age"):
            parts.append(f"Age: {runner['horse_age']}yo")
        if runner.get("barrier"):
            parts.append(f"Barrier: {runner['barrier']}")
        if runner.get("jockey"):
            parts.append(f"Jockey: {runner['jockey']}")
        if runner.get("trainer"):
            parts.append(f"Trainer: {runner['trainer']}")

        # Market signals
        if runner.get("odds_movement"):
            parts.append(f"Market: {runner['odds_movement']}")
        if runner.get("current_odds"):
            parts.append(f"Odds: ${runner['current_odds']:.2f}")

        # Pace dynamics
        if runner.get("speed_map_position"):
            parts.append(f"Map: {runner['speed_map_position']}")
        if runner.get("pf_map_factor"):
            parts.append(f"Pace advantage: {runner['pf_map_factor']:.2f}")

        # First/second up
        if runner.get("days_since_last_run"):
            days = runner["days_since_last_run"]
            if days > 60:
                parts.append("First-up")
            elif 14 < days <= 35:
                parts.append("Second-up")

        # Track/distance form
        if runner.get("track_dist_stats"):
            parts.append(f"Track/Dist: {runner['track_dist_stats']}")

        return " | ".join(parts) if parts else "No context available"

    async def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using OpenAI's embedding model."""
        client = await self._get_client()
        if not client:
            return None

        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None

    async def embed_context(
        self, context: dict[str, Any], runner: dict[str, Any]
    ) -> list[float] | None:
        """Generate embedding for a race context and runner combination."""
        text = self.build_context_text(context, runner)
        return await self.get_embedding(text)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def find_similar(
        self,
        query_embedding: list[float],
        memories: list[tuple[int, list[float]]],  # [(id, embedding), ...]
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> list[tuple[int, float]]:
        """Find most similar memories to the query.

        Returns: [(memory_id, similarity_score), ...]
        """
        if not query_embedding or not memories:
            return []

        scored = []
        for mem_id, embedding in memories:
            if embedding:
                score = self.cosine_similarity(query_embedding, embedding)
                if score >= min_similarity:
                    scored.append((mem_id, score))

        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
