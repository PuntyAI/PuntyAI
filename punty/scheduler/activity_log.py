"""Activity log for tracking scheduler and system actions."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

from punty.config import melb_now_naive

logger = logging.getLogger(__name__)

# Activity types
class ActivityType:
    SCRAPE_START = "scrape_start"
    SCRAPE_COMPLETE = "scrape_complete"
    SCRAPE_ERROR = "scrape_error"
    GENERATE_START = "generate_start"
    GENERATE_COMPLETE = "generate_complete"
    GENERATE_ERROR = "generate_error"
    SCHEDULER_JOB = "scheduler_job"
    RESULTS_CHECK = "results_check"
    SETTLEMENT = "settlement"
    SYSTEM = "system"


@dataclass
class ActivityEntry:
    """Single activity log entry."""

    timestamp: datetime
    activity_type: str
    message: str
    venue: Optional[str] = None
    details: Optional[str] = None
    status: str = "info"  # info, success, warning, error

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_str": self.timestamp.strftime("%H:%M:%S"),
            "activity_type": self.activity_type,
            "message": self.message,
            "venue": self.venue,
            "details": self.details,
            "status": self.status,
        }


class ActivityLog:
    """In-memory activity log with fixed size."""

    def __init__(self, max_entries: int = 100):
        self._entries: deque[ActivityEntry] = deque(maxlen=max_entries)

    def log(
        self,
        activity_type: str,
        message: str,
        venue: Optional[str] = None,
        details: Optional[str] = None,
        status: str = "info",
    ) -> None:
        """Add an activity to the log."""
        entry = ActivityEntry(
            timestamp=melb_now_naive(),
            activity_type=activity_type,
            message=message,
            venue=venue,
            details=details,
            status=status,
        )
        self._entries.appendleft(entry)

        # Also log to standard logger
        log_level = logging.INFO
        if status == "error":
            log_level = logging.ERROR
        elif status == "warning":
            log_level = logging.WARNING
        logger.log(log_level, f"[Activity] {message}" + (f" ({venue})" if venue else ""))

    def get_entries(self, limit: int = 50) -> list[dict]:
        """Get recent entries as dicts."""
        entries = list(self._entries)[:limit]
        return [e.to_dict() for e in entries]

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()


# Global activity log instance
activity_log = ActivityLog()


# Convenience functions
def log_scrape_start(venue: str, action: str = "Scraping") -> None:
    """Log start of a scrape operation."""
    activity_log.log(
        ActivityType.SCRAPE_START,
        f"{action} {venue}",
        venue=venue,
        status="info",
    )


def log_scrape_complete(venue: str, details: Optional[str] = None) -> None:
    """Log successful scrape completion."""
    activity_log.log(
        ActivityType.SCRAPE_COMPLETE,
        f"Scraped {venue}",
        venue=venue,
        details=details,
        status="success",
    )


def log_scrape_error(venue: str, error: str) -> None:
    """Log scrape error."""
    activity_log.log(
        ActivityType.SCRAPE_ERROR,
        f"Scrape failed: {venue}",
        venue=venue,
        details=error,
        status="error",
    )


def log_generate_start(venue: str, content_type: str = "Early Mail") -> None:
    """Log start of content generation."""
    activity_log.log(
        ActivityType.GENERATE_START,
        f"Generating {content_type} for {venue}",
        venue=venue,
        status="info",
    )


def log_generate_complete(venue: str, content_type: str = "Early Mail") -> None:
    """Log successful content generation."""
    activity_log.log(
        ActivityType.GENERATE_COMPLETE,
        f"Generated {content_type} for {venue}",
        venue=venue,
        status="success",
    )


def log_generate_error(venue: str, error: str, content_type: str = "Early Mail") -> None:
    """Log content generation error."""
    activity_log.log(
        ActivityType.GENERATE_ERROR,
        f"{content_type} generation failed: {venue}",
        venue=venue,
        details=error,
        status="error",
    )


def log_scheduler_job(job_name: str, status: str = "info") -> None:
    """Log scheduler job execution."""
    activity_log.log(
        ActivityType.SCHEDULER_JOB,
        f"Scheduler: {job_name}",
        status=status,
    )


def log_results_check(venue: str, races_checked: int = 0) -> None:
    """Log results check."""
    activity_log.log(
        ActivityType.RESULTS_CHECK,
        f"Checked results: {venue}",
        venue=venue,
        details=f"{races_checked} races" if races_checked else None,
        status="info",
    )


def log_settlement(venue: str, picks_settled: int = 0, pnl: float = 0) -> None:
    """Log settlement."""
    status = "success" if pnl >= 0 else "warning"
    activity_log.log(
        ActivityType.SETTLEMENT,
        f"Settled {picks_settled} picks for {venue}",
        venue=venue,
        details=f"P&L: ${pnl:+.2f}" if pnl else None,
        status=status,
    )


def log_system(message: str, status: str = "info") -> None:
    """Log system event."""
    activity_log.log(
        ActivityType.SYSTEM,
        message,
        status=status,
    )
