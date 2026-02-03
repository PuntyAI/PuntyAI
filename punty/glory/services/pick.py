"""Pick submission service for Group One Glory."""

import uuid
from typing import Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.config import melb_now
from punty.models.glory import G1Pick, G1Race, G1Horse, G1User

# Points for finishing positions
POINTS_1ST = 3
POINTS_2ND = 2
POINTS_3RD = 1


def generate_pick_id() -> str:
    """Generate a unique pick ID."""
    return f"g1p_{uuid.uuid4().hex[:16]}"


class PickService:
    """Service for managing user picks."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def submit_pick(
        self,
        user_id: str,
        race_id: str,
        horse_id: str,
    ) -> G1Pick:
        """Submit or update a pick for a race.

        If the user already has a pick for this race, it will be updated.
        """
        # Check if race is open for tipping
        race = await self._get_race(race_id)
        if not race:
            raise ValueError("Race not found")

        if race.status != "open":
            raise ValueError("Race is not open for tipping")

        if race.tipping_closes_at:
            now = melb_now()
            closes = race.tipping_closes_at
            if closes.tzinfo is None:
                closes = closes.replace(tzinfo=now.tzinfo)
            if now > closes:
                raise ValueError("Tipping has closed for this race")

        # Verify horse exists and is not scratched
        horse = await self._get_horse(horse_id)
        if not horse:
            raise ValueError("Horse not found")

        if horse.race_id != race_id:
            raise ValueError("Horse is not in this race")

        if horse.is_scratched:
            raise ValueError("Cannot pick a scratched horse")

        # Check for existing pick
        existing = await self.get_user_pick_for_race(user_id, race_id)

        if existing:
            # Update existing pick
            existing.horse_id = horse_id
            existing.points_earned = None  # Reset points if race not yet resulted
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
        else:
            # Create new pick
            pick = G1Pick(
                id=generate_pick_id(),
                user_id=user_id,
                race_id=race_id,
                horse_id=horse_id,
            )
            self.db.add(pick)
            await self.db.commit()
            await self.db.refresh(pick)
            return pick

    async def get_user_pick_for_race(
        self, user_id: str, race_id: str
    ) -> Optional[G1Pick]:
        """Get a user's pick for a specific race."""
        result = await self.db.execute(
            select(G1Pick)
            .where(G1Pick.user_id == user_id)
            .where(G1Pick.race_id == race_id)
            .options(selectinload(G1Pick.horse))
        )
        return result.scalar_one_or_none()

    async def get_user_picks_for_competition(
        self, user_id: str, competition_id: str
    ) -> list[G1Pick]:
        """Get all picks for a user in a competition."""
        result = await self.db.execute(
            select(G1Pick)
            .join(G1Race)
            .where(G1Pick.user_id == user_id)
            .where(G1Race.competition_id == competition_id)
            .options(selectinload(G1Pick.horse), selectinload(G1Pick.race))
            .order_by(G1Race.race_date)
        )
        return list(result.scalars().all())

    async def get_all_picks_for_race(self, race_id: str) -> list[G1Pick]:
        """Get all picks for a race (admin view)."""
        result = await self.db.execute(
            select(G1Pick)
            .where(G1Pick.race_id == race_id)
            .options(
                selectinload(G1Pick.horse),
                selectinload(G1Pick.user),
            )
        )
        return list(result.scalars().all())

    async def delete_pick(self, user_id: str, race_id: str) -> bool:
        """Delete a user's pick for a race (if race is still open)."""
        race = await self._get_race(race_id)
        if not race or race.status != "open":
            return False

        result = await self.db.execute(
            delete(G1Pick)
            .where(G1Pick.user_id == user_id)
            .where(G1Pick.race_id == race_id)
        )
        await self.db.commit()
        return result.rowcount > 0

    async def calculate_points_for_race(self, race_id: str) -> dict[str, int]:
        """Calculate points for all picks in a race.

        Returns dict mapping user_id to points earned.
        """
        # Get race with horses
        race = await self._get_race(race_id)
        if not race or race.status != "resulted":
            return {}

        # Get horses with finish positions
        result = await self.db.execute(
            select(G1Horse)
            .where(G1Horse.race_id == race_id)
            .where(G1Horse.finish_position.isnot(None))
        )
        horses = result.scalars().all()

        # Build position lookup
        horse_positions = {h.id: h.finish_position for h in horses}

        # Get all picks for this race
        picks = await self.get_all_picks_for_race(race_id)

        # Calculate points
        user_points = {}
        for pick in picks:
            position = horse_positions.get(pick.horse_id)
            if position == 1:
                points = POINTS_1ST
            elif position == 2:
                points = POINTS_2ND
            elif position == 3:
                points = POINTS_3RD
            else:
                points = 0

            # Update pick with points
            pick.points_earned = points
            user_points[pick.user_id] = points

        await self.db.commit()
        return user_points

    async def get_user_total_points(
        self, user_id: str, competition_id: str
    ) -> int:
        """Get a user's total points in a competition."""
        picks = await self.get_user_picks_for_competition(user_id, competition_id)
        return sum(p.points_earned or 0 for p in picks)

    async def get_pick_summary_for_user(
        self, user_id: str, competition_id: str
    ) -> dict:
        """Get a summary of a user's picks in a competition."""
        picks = await self.get_user_picks_for_competition(user_id, competition_id)

        total_picks = len(picks)
        total_points = sum(p.points_earned or 0 for p in picks)
        winners = sum(1 for p in picks if p.points_earned == POINTS_1ST)
        seconds = sum(1 for p in picks if p.points_earned == POINTS_2ND)
        thirds = sum(1 for p in picks if p.points_earned == POINTS_3RD)
        pending = sum(1 for p in picks if p.points_earned is None)

        return {
            "total_picks": total_picks,
            "total_points": total_points,
            "winners": winners,
            "seconds": seconds,
            "thirds": thirds,
            "pending_results": pending,
        }

    async def _get_race(self, race_id: str) -> Optional[G1Race]:
        """Get a race by ID."""
        result = await self.db.execute(
            select(G1Race).where(G1Race.id == race_id)
        )
        return result.scalar_one_or_none()

    async def _get_horse(self, horse_id: str) -> Optional[G1Horse]:
        """Get a horse by ID."""
        result = await self.db.execute(
            select(G1Horse).where(G1Horse.id == horse_id)
        )
        return result.scalar_one_or_none()
