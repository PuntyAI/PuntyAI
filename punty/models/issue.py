"""Persistent issue tracker for data quality and settlement problems."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, String, Text, Boolean, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from punty.config import melb_now_naive
from punty.models.database import Base


class Issue(Base):
    __tablename__ = "issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(50))  # settlement, scrape, exotic, betfair, data
    severity: Mapped[str] = mapped_column(String(20))  # error, warning, info
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meeting_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    race_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pick_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    link: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # URL to fix
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # $ impact
    created_at: Mapped[datetime] = mapped_column(DateTime, default=melb_now_naive)
