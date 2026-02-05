"""Memory store for managing race predictions and pattern learning."""

import json
import logging
from typing import Any, Optional

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.memory.models import RaceMemory, PatternInsight
from punty.memory.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def _escape_like_pattern(s: str) -> str:
    """Escape SQL LIKE special characters to prevent injection."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class MemoryStore:
    """Store and retrieve race prediction memories for pattern learning."""

    def __init__(self, db: AsyncSession, embedding_service: Optional[EmbeddingService] = None):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()

    async def store_prediction(
        self,
        meeting_id: str,
        race_number: int,
        race_context: dict[str, Any],
        runner: dict[str, Any],
        tip_rank: int,
        confidence: str | None = None,
        odds_at_tip: float | None = None,
        bet_type: str | None = None,
        generate_embedding: bool = True,
    ) -> RaceMemory:
        """Store a prediction for later learning.

        Args:
            meeting_id: Meeting ID
            race_number: Race number
            race_context: Dict with track condition, distance, class, etc.
            runner: Dict with form, odds, barrier, jockey, etc.
            tip_rank: 1=top pick, 2, 3, or 4=roughie
            confidence: high/med/low
            odds_at_tip: Odds at time of tip
            bet_type: win/place/each_way
            generate_embedding: Whether to generate embedding for similarity search
        """
        race_id = f"{meeting_id}-r{race_number}"

        memory = RaceMemory(
            meeting_id=meeting_id,
            race_number=race_number,
            race_id=race_id,
            horse_name=runner.get("horse_name", "Unknown"),
            saddlecloth=runner.get("saddlecloth", 0),
            tip_rank=tip_rank,
            confidence=confidence,
            odds_at_tip=odds_at_tip,
            bet_type=bet_type,
        )
        memory.context = race_context
        memory.runner = runner

        # Generate embedding for similarity search
        if generate_embedding:
            try:
                embedding = await self.embedding_service.embed_context(race_context, runner)
                if embedding:
                    memory.embedding = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        self.db.add(memory)
        await self.db.flush()
        return memory

    async def update_outcome(
        self,
        race_id: str,
        saddlecloth: int,
        finish_position: int,
        hit: bool,
        pnl: float,
        sp_odds: float | None = None,
    ) -> Optional[RaceMemory]:
        """Update a memory with the race outcome.

        Called after race results are known.
        """
        result = await self.db.execute(
            select(RaceMemory).where(
                and_(
                    RaceMemory.race_id == race_id,
                    RaceMemory.saddlecloth == saddlecloth,
                )
            )
        )
        memory = result.scalar_one_or_none()

        if memory:
            memory.finish_position = finish_position
            memory.hit = hit
            memory.pnl = pnl
            memory.sp_odds = sp_odds
            memory.settled_at = melb_now_naive()
            await self.db.flush()

        return memory

    async def find_similar_situations(
        self,
        race_context: dict[str, Any],
        runner: dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.7,
        only_settled: bool = True,
    ) -> list[dict[str, Any]]:
        """Find similar past situations for learning.

        Returns memories with outcomes that can inform the current prediction.
        """
        # Generate embedding for current context
        query_embedding = await self.embedding_service.embed_context(race_context, runner)
        if not query_embedding:
            # Fall back to rule-based matching
            return await self._find_similar_by_rules(race_context, runner, top_k)

        # Get all memories with embeddings
        query = select(RaceMemory).where(RaceMemory.embedding_json.isnot(None))
        if only_settled:
            query = query.where(RaceMemory.settled_at.isnot(None))

        result = await self.db.execute(query)
        memories = result.scalars().all()

        # Calculate similarities
        memory_embeddings = [
            (m.id, m.embedding) for m in memories if m.embedding
        ]
        similar_ids = self.embedding_service.find_similar(
            query_embedding, memory_embeddings, top_k, min_similarity
        )

        # Fetch full memory details for similar ones
        similar_memories = []
        for mem_id, score in similar_ids:
            for m in memories:
                if m.id == mem_id:
                    mem_dict = m.to_dict()
                    mem_dict["similarity_score"] = score
                    similar_memories.append(mem_dict)
                    break

        return similar_memories

    async def _find_similar_by_rules(
        self,
        race_context: dict[str, Any],
        runner: dict[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Fallback: find similar situations using rule-based matching."""
        conditions = []

        # Match on track condition
        track = race_context.get("track_condition", "").lower()
        if "heavy" in track:
            conditions.append("heavy")
        elif "soft" in track:
            conditions.append("soft")
        else:
            conditions.append("good")

        # Match on distance range
        dist = race_context.get("distance", 1200)
        if dist <= 1200:
            dist_range = "sprint"
        elif dist <= 1600:
            dist_range = "mile"
        else:
            dist_range = "staying"

        # Match on market movement
        movement = runner.get("odds_movement", "stable")

        # Query for similar settled memories
        # This is a simplified rule-based approach
        query = (
            select(RaceMemory)
            .where(RaceMemory.settled_at.isnot(None))
            .order_by(desc(RaceMemory.created_at))
            .limit(top_k * 3)  # Get more and filter
        )
        result = await self.db.execute(query)
        memories = result.scalars().all()

        # Score and filter
        scored = []
        for m in memories:
            ctx = m.context
            rnr = m.runner
            score = 0

            # Track condition match
            m_track = ctx.get("track_condition", "").lower()
            if ("heavy" in m_track and "heavy" in track) or \
               ("soft" in m_track and "soft" in track) or \
               ("good" in m_track and "good" in track):
                score += 2

            # Distance range match
            m_dist = ctx.get("distance", 1200)
            if (dist <= 1200 and m_dist <= 1200) or \
               (1200 < dist <= 1600 and 1200 < m_dist <= 1600) or \
               (dist > 1600 and m_dist > 1600):
                score += 2

            # Market movement match
            m_movement = rnr.get("odds_movement", "stable")
            if m_movement == movement:
                score += 1

            # Tip rank match
            if m.tip_rank == runner.get("tip_rank", 1):
                score += 1

            if score >= 2:
                mem_dict = m.to_dict()
                mem_dict["similarity_score"] = score / 6.0  # Normalize
                scored.append((score, mem_dict))

        # Sort by score and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def get_pattern_insights(
        self,
        race_context: dict[str, Any],
        runner: dict[str, Any],
    ) -> list[str]:
        """Get relevant pattern insights for the current situation.

        Returns human-readable insights that can be included in prompts.
        """
        insights = []

        # Track condition pattern
        track = race_context.get("track_condition", "").lower()
        if "heavy" in track:
            pattern_key = f"heavy_{race_context.get('distance', 1200)}"
        elif "soft" in track:
            pattern_key = f"soft_{race_context.get('distance', 1200)}"
        else:
            pattern_key = None

        if pattern_key:
            safe_pattern = _escape_like_pattern(pattern_key[:10])
            result = await self.db.execute(
                select(PatternInsight).where(
                    PatternInsight.pattern_key.like(f"%{safe_pattern}%", escape="\\")
                ).limit(3)
            )
            patterns = result.scalars().all()
            for p in patterns:
                if p.sample_count >= 10:  # Only use well-supported patterns
                    insights.append(p.insight_text)

        # Market movement patterns
        movement = runner.get("odds_movement", "stable")
        if movement in ("heavy_support", "firming"):
            safe_movement = _escape_like_pattern(movement)
            result = await self.db.execute(
                select(PatternInsight).where(
                    PatternInsight.pattern_type == "market_move",
                    PatternInsight.pattern_key.like(f"%{safe_movement}%", escape="\\"),
                ).limit(2)
            )
            patterns = result.scalars().all()
            for p in patterns:
                if p.sample_count >= 5:
                    insights.append(p.insight_text)

        return insights

    async def build_learning_context(
        self,
        race_context: dict[str, Any],
        runner: dict[str, Any],
        max_memories: int = 3,
    ) -> str:
        """Build a learning context string for inclusion in AI prompts.

        Returns a formatted string with:
        - Similar past situations and their outcomes
        - Pattern insights that apply
        """
        parts = []

        # Find similar past situations
        similar = await self.find_similar_situations(
            race_context, runner, top_k=max_memories, min_similarity=0.65
        )

        if similar:
            parts.append("**Similar Past Situations:**")
            for mem in similar:
                outcome = "✓" if mem.get("hit") else "✗"
                pnl = mem.get("pnl", 0)
                pnl_str = f"+{pnl:.1f}U" if pnl > 0 else f"{pnl:.1f}U"
                ctx = mem.get("context", {})
                rnr = mem.get("runner", {})

                line = (
                    f"- {mem['horse_name']} (Rank {mem['tip_rank']}) @ "
                    f"${mem.get('odds_at_tip', 0):.2f}: "
                    f"Finished {mem.get('finish_position', '?')} {outcome} ({pnl_str})"
                )
                # Add key context
                conditions = []
                if ctx.get("track_condition"):
                    conditions.append(ctx["track_condition"])
                if ctx.get("distance"):
                    conditions.append(f"{ctx['distance']}m")
                if rnr.get("odds_movement"):
                    conditions.append(f"Market: {rnr['odds_movement']}")
                if conditions:
                    line += f" | {', '.join(conditions)}"

                parts.append(line)

        # Get pattern insights
        insights = await self.get_pattern_insights(race_context, runner)
        if insights:
            parts.append("")
            parts.append("**Pattern Insights:**")
            for insight in insights[:3]:
                parts.append(f"- {insight}")

        return "\n".join(parts) if parts else ""

    async def get_stats(self) -> dict[str, Any]:
        """Get overall memory statistics."""
        # Total memories
        total_result = await self.db.execute(select(func.count(RaceMemory.id)))
        total = total_result.scalar_one()

        # Settled memories
        settled_result = await self.db.execute(
            select(func.count(RaceMemory.id)).where(RaceMemory.settled_at.isnot(None))
        )
        settled = settled_result.scalar_one()

        # Hit rate
        if settled > 0:
            hits_result = await self.db.execute(
                select(func.count(RaceMemory.id)).where(
                    RaceMemory.settled_at.isnot(None),
                    RaceMemory.hit == True,
                )
            )
            hits = hits_result.scalar_one()
            hit_rate = hits / settled * 100
        else:
            hit_rate = 0.0

        # Average PNL
        pnl_result = await self.db.execute(
            select(func.avg(RaceMemory.pnl)).where(RaceMemory.settled_at.isnot(None))
        )
        avg_pnl = pnl_result.scalar_one() or 0.0

        return {
            "total_memories": total,
            "settled_memories": settled,
            "hit_rate": hit_rate,
            "avg_pnl": avg_pnl,
        }
