"""Race management service for Group One Glory."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from punty.config import melb_now
from punty.models.glory import G1Race, G1Horse, G1Pick


def generate_race_id() -> str:
    """Generate a unique race ID."""
    return f"g1r_{uuid.uuid4().hex[:16]}"


def generate_horse_id() -> str:
    """Generate a unique horse ID."""
    return f"g1h_{uuid.uuid4().hex[:16]}"


class RaceService:
    """Service for managing Group 1 races."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_race(
        self,
        competition_id: str,
        race_name: str,
        venue: str,
        race_date: datetime,
        distance: int,
        prize_money: Optional[int] = None,
        external_id: Optional[str] = None,
        race_number: Optional[int] = None,
    ) -> G1Race:
        """Create a new race."""
        race = G1Race(
            id=generate_race_id(),
            competition_id=competition_id,
            external_id=external_id,
            race_name=race_name,
            venue=venue,
            race_date=race_date,
            distance=distance,
            prize_money=prize_money,
            race_number=race_number,
            status="nominations",
        )
        self.db.add(race)
        await self.db.commit()
        await self.db.refresh(race)
        return race

    async def get_race(
        self, race_id: str, include_horses: bool = False
    ) -> Optional[G1Race]:
        """Get a race by ID."""
        query = select(G1Race).where(G1Race.id == race_id)
        if include_horses:
            query = query.options(selectinload(G1Race.horses))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_races_by_competition(
        self, competition_id: str, include_horses: bool = False
    ) -> list[G1Race]:
        """List all races in a competition, ordered by date."""
        query = (
            select(G1Race)
            .where(G1Race.competition_id == competition_id)
            .order_by(G1Race.race_date)
        )
        if include_horses:
            query = query.options(selectinload(G1Race.horses))
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def list_open_races(self, competition_id: str) -> list[G1Race]:
        """List races that are currently open for tipping."""
        query = (
            select(G1Race)
            .where(G1Race.competition_id == competition_id)
            .where(G1Race.status == "open")
            .order_by(G1Race.race_date)
            .options(selectinload(G1Race.horses))
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update_race_status(
        self,
        race_id: str,
        status: str,
        tipping_closes_at: Optional[datetime] = None,
    ) -> Optional[G1Race]:
        """Update a race's status."""
        race = await self.get_race(race_id)
        if not race:
            return None

        race.status = status
        if tipping_closes_at is not None:
            race.tipping_closes_at = tipping_closes_at

        await self.db.commit()
        await self.db.refresh(race)
        return race

    async def is_tipping_open(self, race_id: str) -> bool:
        """Check if tipping is currently open for a race."""
        race = await self.get_race(race_id)
        if not race:
            return False

        if race.status != "open":
            return False

        if race.tipping_closes_at and melb_now() > race.tipping_closes_at.replace(tzinfo=melb_now().tzinfo):
            return False

        return True

    async def add_horse(
        self,
        race_id: str,
        name: str,
        barrier: Optional[int] = None,
        weight: Optional[float] = None,
        jockey: Optional[str] = None,
        trainer: Optional[str] = None,
        odds: Optional[float] = None,
        form: Optional[str] = None,
        rating: Optional[int] = None,
        career_record: Optional[str] = None,
        career_prize: Optional[int] = None,
        saddlecloth: Optional[int] = None,
        horse_age: Optional[int] = None,
        horse_sex: Optional[str] = None,
        sire: Optional[str] = None,
        dam: Optional[str] = None,
        last_five: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> G1Horse:
        """Add a horse to a race."""
        horse = G1Horse(
            id=generate_horse_id(),
            race_id=race_id,
            name=name,
            barrier=barrier,
            weight=weight,
            jockey=jockey,
            trainer=trainer,
            odds=odds,
            form=form,
            rating=rating,
            career_record=career_record,
            career_prize=career_prize,
            saddlecloth=saddlecloth,
            horse_age=horse_age,
            horse_sex=horse_sex,
            sire=sire,
            dam=dam,
            last_five=last_five,
            comment=comment,
        )
        self.db.add(horse)
        await self.db.commit()
        await self.db.refresh(horse)
        return horse

    async def get_horse(self, horse_id: str) -> Optional[G1Horse]:
        """Get a horse by ID."""
        result = await self.db.execute(
            select(G1Horse).where(G1Horse.id == horse_id)
        )
        return result.scalar_one_or_none()

    async def list_horses(self, race_id: str, exclude_scratched: bool = True) -> list[G1Horse]:
        """List horses in a race."""
        query = select(G1Horse).where(G1Horse.race_id == race_id)
        if exclude_scratched:
            query = query.where(G1Horse.is_scratched == False)
        query = query.order_by(G1Horse.saddlecloth, G1Horse.barrier, G1Horse.name)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def scratch_horse(
        self, horse_id: str, reason: Optional[str] = None
    ) -> Optional[G1Horse]:
        """Mark a horse as scratched."""
        horse = await self.get_horse(horse_id)
        if not horse:
            return None

        horse.is_scratched = True
        horse.scratching_reason = reason
        await self.db.commit()
        await self.db.refresh(horse)
        return horse

    async def update_horse(
        self,
        horse_id: str,
        **kwargs,
    ) -> Optional[G1Horse]:
        """Update horse details."""
        horse = await self.get_horse(horse_id)
        if not horse:
            return None

        for key, value in kwargs.items():
            if hasattr(horse, key) and value is not None:
                setattr(horse, key, value)

        await self.db.commit()
        await self.db.refresh(horse)
        return horse

    async def clear_horses(self, race_id: str) -> int:
        """Remove all horses from a race (for re-import)."""
        # First delete picks that reference these horses
        horses = await self.list_horses(race_id, exclude_scratched=False)
        horse_ids = [h.id for h in horses]

        if horse_ids:
            await self.db.execute(
                G1Pick.__table__.delete().where(G1Pick.horse_id.in_(horse_ids))
            )

        # Now delete horses
        result = await self.db.execute(
            G1Horse.__table__.delete().where(G1Horse.race_id == race_id)
        )
        await self.db.commit()
        return result.rowcount

    async def enter_results(
        self,
        race_id: str,
        results: list[tuple[str, int]],  # List of (horse_id, position)
    ) -> G1Race:
        """Enter race results and update horse finish positions."""
        race = await self.get_race(race_id, include_horses=True)
        if not race:
            raise ValueError(f"Race {race_id} not found")

        # Update finish positions
        for horse_id, position in results:
            await self.db.execute(
                update(G1Horse)
                .where(G1Horse.id == horse_id)
                .values(finish_position=position)
            )

        # Update race status
        race.status = "resulted"
        await self.db.commit()
        await self.db.refresh(race)
        return race

    async def delete_race(self, race_id: str) -> bool:
        """Delete a race and all its horses."""
        race = await self.get_race(race_id)
        if not race:
            return False

        await self.db.delete(race)
        await self.db.commit()
        return True

    async def get_races_grouped_by_date(
        self, competition_id: str
    ) -> dict[str, list[G1Race]]:
        """Get races grouped by date for display."""
        races = await self.list_races_by_competition(competition_id, include_horses=True)

        grouped: dict[str, list[G1Race]] = {}
        for race in races:
            date_key = race.race_date.strftime("%Y-%m-%d")
            if date_key not in grouped:
                grouped[date_key] = []
            grouped[date_key].append(race)

        return grouped
