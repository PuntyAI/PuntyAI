"""Competition management service for Group One Glory."""

import uuid
from datetime import date
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.models.glory import G1Competition, G1Race


def generate_competition_id() -> str:
    """Generate a unique competition ID."""
    return f"g1c_{uuid.uuid4().hex[:16]}"


class CompetitionService:
    """Service for managing tipping competitions."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_competition(
        self,
        name: str,
        start_date: date,
        end_date: date,
        prize_pool: Optional[int] = None,
        is_active: bool = False,
    ) -> G1Competition:
        """Create a new competition."""
        competition = G1Competition(
            id=generate_competition_id(),
            name=name,
            start_date=start_date,
            end_date=end_date,
            prize_pool=prize_pool,
            is_active=is_active,
        )
        self.db.add(competition)
        await self.db.commit()
        await self.db.refresh(competition)
        return competition

    async def get_competition(
        self, competition_id: str, include_races: bool = False
    ) -> Optional[G1Competition]:
        """Get a competition by ID."""
        query = select(G1Competition).where(G1Competition.id == competition_id)
        if include_races:
            query = query.options(
                selectinload(G1Competition.races).selectinload(G1Race.horses)
            )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_active_competition(
        self, include_races: bool = False
    ) -> Optional[G1Competition]:
        """Get the currently active competition."""
        query = select(G1Competition).where(G1Competition.is_active == True)
        if include_races:
            query = query.options(
                selectinload(G1Competition.races).selectinload(G1Race.horses)
            )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_competitions(
        self, include_races: bool = False
    ) -> list[G1Competition]:
        """List all competitions, ordered by start date descending."""
        query = select(G1Competition).order_by(G1Competition.start_date.desc())
        if include_races:
            query = query.options(
                selectinload(G1Competition.races).selectinload(G1Race.horses)
            )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def activate_competition(self, competition_id: str) -> G1Competition:
        """Activate a competition (deactivates any other active competition)."""
        # Deactivate all other competitions
        await self.db.execute(
            update(G1Competition)
            .where(G1Competition.id != competition_id)
            .values(is_active=False)
        )

        # Activate the specified competition
        await self.db.execute(
            update(G1Competition)
            .where(G1Competition.id == competition_id)
            .values(is_active=True)
        )
        await self.db.commit()

        return await self.get_competition(competition_id)

    async def deactivate_competition(self, competition_id: str) -> G1Competition:
        """Deactivate a competition."""
        await self.db.execute(
            update(G1Competition)
            .where(G1Competition.id == competition_id)
            .values(is_active=False)
        )
        await self.db.commit()
        return await self.get_competition(competition_id)

    async def update_competition(
        self,
        competition_id: str,
        name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        prize_pool: Optional[int] = None,
    ) -> Optional[G1Competition]:
        """Update a competition's details."""
        competition = await self.get_competition(competition_id)
        if not competition:
            return None

        if name is not None:
            competition.name = name
        if start_date is not None:
            competition.start_date = start_date
        if end_date is not None:
            competition.end_date = end_date
        if prize_pool is not None:
            competition.prize_pool = prize_pool

        await self.db.commit()
        await self.db.refresh(competition)
        return competition

    async def delete_competition(self, competition_id: str) -> bool:
        """Delete a competition and all its races."""
        competition = await self.get_competition(competition_id)
        if not competition:
            return False

        await self.db.delete(competition)
        await self.db.commit()
        return True

    async def get_competition_stats(self, competition_id: str) -> dict:
        """Get statistics for a competition."""
        competition = await self.get_competition(competition_id, include_races=True)
        if not competition:
            return {}

        total_races = len(competition.races)
        resulted_races = sum(1 for r in competition.races if r.status == "resulted")
        open_races = sum(1 for r in competition.races if r.status == "open")
        upcoming_races = sum(
            1 for r in competition.races if r.status in ("nominations", "final_field")
        )

        return {
            "total_races": total_races,
            "resulted_races": resulted_races,
            "open_races": open_races,
            "upcoming_races": upcoming_races,
            "completion_percent": (
                round(resulted_races / total_races * 100) if total_races > 0 else 0
            ),
        }
