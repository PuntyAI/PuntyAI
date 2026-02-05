"""Leaderboard calculation service for Group One Glory."""

import uuid
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.models.glory import G1Pick, G1Race, G1User, G1Competition, G1HonorBoard


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""

    rank: int
    user_id: str
    display_name: str
    total_points: int
    races_picked: int
    winners: int
    places: int  # 2nd and 3rd combined
    points_behind: int = 0


def generate_honor_id() -> str:
    """Generate a unique honor board ID."""
    return f"g1hb_{uuid.uuid4().hex[:16]}"


class LeaderboardService:
    """Service for calculating and managing leaderboards."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_leaderboard(
        self,
        competition_id: str,
        limit: Optional[int] = None,
    ) -> list[LeaderboardEntry]:
        """Calculate the leaderboard for a competition.

        Returns list of LeaderboardEntry sorted by points (desc), then winners (desc).
        Includes all users who have made picks, even if no races have resulted yet.
        """
        # Get all races in competition
        races_result = await self.db.execute(
            select(G1Race.id)
            .where(G1Race.competition_id == competition_id)
        )
        all_race_ids = [r[0] for r in races_result.all()]

        if not all_race_ids:
            return []

        # Get resulted races for points calculation
        resulted_result = await self.db.execute(
            select(G1Race.id)
            .where(G1Race.competition_id == competition_id)
            .where(G1Race.status == "resulted")
        )
        resulted_race_ids = [r[0] for r in resulted_result.all()]

        # Aggregate picks by user - include all users who have made ANY picks
        # Build case expressions for resulted races only
        points_case = case(
            (G1Pick.race_id.in_(resulted_race_ids), func.coalesce(G1Pick.points_earned, 0)),
            else_=0
        ) if resulted_race_ids else 0

        winners_case = case(
            (and_(G1Pick.race_id.in_(resulted_race_ids), G1Pick.points_earned == 3), 1),
            else_=0
        ) if resulted_race_ids else 0

        places_case = case(
            (and_(
                G1Pick.race_id.in_(resulted_race_ids),
                (G1Pick.points_earned == 2) | (G1Pick.points_earned == 1)
            ), 1),
            else_=0
        ) if resulted_race_ids else 0

        query = (
            select(
                G1Pick.user_id,
                G1User.display_name,
                func.sum(points_case).label("total_points"),
                func.count(G1Pick.id).label("races_picked"),
                func.sum(winners_case).label("winners"),
                func.sum(places_case).label("places"),
            )
            .join(G1User)
            .where(G1Pick.race_id.in_(all_race_ids))
            .group_by(G1Pick.user_id, G1User.display_name)
            .order_by(
                func.sum(points_case).desc(),
                func.sum(winners_case).desc(),
                G1User.display_name.asc(),  # Alphabetical for ties
            )
        )

        if limit:
            query = query.limit(limit)

        result = await self.db.execute(query)
        rows = result.all()

        # Build leaderboard entries
        entries = []
        leader_points = 0

        for i, row in enumerate(rows):
            if i == 0:
                leader_points = row.total_points or 0

            entry = LeaderboardEntry(
                rank=i + 1,
                user_id=row.user_id,
                display_name=row.display_name,
                total_points=row.total_points or 0,
                races_picked=row.races_picked or 0,
                winners=row.winners or 0,
                places=row.places or 0,
                points_behind=leader_points - (row.total_points or 0),
            )
            entries.append(entry)

        return entries

    async def get_user_rank(
        self, user_id: str, competition_id: str
    ) -> Optional[LeaderboardEntry]:
        """Get a specific user's leaderboard entry."""
        leaderboard = await self.get_leaderboard(competition_id)
        for entry in leaderboard:
            if entry.user_id == user_id:
                return entry
        return None

    async def get_top_3(self, competition_id: str) -> list[LeaderboardEntry]:
        """Get the top 3 entries (for podium display)."""
        return await self.get_leaderboard(competition_id, limit=3)

    async def finalize_competition(self, competition_id: str) -> list[G1HonorBoard]:
        """Finalize a competition and create honor board entries.

        Should be called when all races in a competition are resulted.
        """
        # Get final leaderboard
        leaderboard = await self.get_leaderboard(competition_id)

        # Create honor board entries for top participants
        honor_entries = []
        for entry in leaderboard:
            honor = G1HonorBoard(
                id=generate_honor_id(),
                competition_id=competition_id,
                user_id=entry.user_id,
                final_points=entry.total_points,
                final_rank=entry.rank,
            )
            self.db.add(honor)
            honor_entries.append(honor)

        await self.db.commit()
        return honor_entries

    async def get_honor_board(self, limit: int = 20) -> list[dict]:
        """Get the honor board showing past champions."""
        result = await self.db.execute(
            select(G1HonorBoard)
            .where(G1HonorBoard.final_rank == 1)  # Only winners
            .options(
                selectinload(G1HonorBoard.competition),
                selectinload(G1HonorBoard.user),
            )
            .order_by(G1HonorBoard.created_at.desc())
            .limit(limit)
        )
        entries = result.scalars().all()
        return [e.to_dict() for e in entries]

    async def get_competition_summary(self, competition_id: str) -> dict:
        """Get a summary of the competition standings."""
        leaderboard = await self.get_leaderboard(competition_id)

        if not leaderboard:
            return {
                "total_participants": 0,
                "total_points_awarded": 0,
                "leader": None,
                "top_3": [],
            }

        total_points = sum(e.total_points for e in leaderboard)

        return {
            "total_participants": len(leaderboard),
            "total_points_awarded": total_points,
            "leader": {
                "display_name": leaderboard[0].display_name,
                "points": leaderboard[0].total_points,
            } if leaderboard else None,
            "top_3": [
                {
                    "rank": e.rank,
                    "display_name": e.display_name,
                    "points": e.total_points,
                    "winners": e.winners,
                }
                for e in leaderboard[:3]
            ],
        }
