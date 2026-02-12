"""Settings models for configurable analysis weights and preferences."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_now_naive
from punty.models.database import Base




class AnalysisWeights(Base):
    """Configurable weights for the analysis framework."""

    __tablename__ = "analysis_weights"

    id = Column(String, primary_key=True, default="default")
    name = Column(String, default="Default Weights")

    # Weight values stored as JSON
    # Each weight can be: "low", "low-med", "med", "med-high", "high"
    weights_json = Column(Text, default="{}")

    created_at = Column(DateTime, default=melb_now_naive)
    updated_at = Column(DateTime, default=melb_now_naive, onupdate=melb_now_naive)

    # Default weights from PUNTY_MASTER v3.13
    DEFAULT_WEIGHTS = {
        "recent_form": "high",
        "class": "high",
        "track_conditions": "high",
        "distance_fit": "high",
        "barrier_draw": "high",
        "jockey": "high",
        "trainer": "high",
        "speed_map_race_shape": "high",
        "weight": "high",
        "sectionals": "high",
        "barrier_track_bias": "high",
        "head_to_head": "med",
        "first_second_up": "med-high",
        "course": "med-high",
        "gear": "med",
        "market": "med-high",
        "trials": "med-high",
        "pedigree": "low-med",
        "money": "low-med",
        "tipsters_analysis": "high",
        # Pace analysis insights
        "pf_map_factor": "med",  # Pace advantage/disadvantage factor
        "pf_speed_rank": "med",  # Early speed rating (1=fastest)
        "pf_settle_position": "med",  # Historical settling position
        "pf_jockey_factor": "med",  # Jockey effectiveness metric
        # Additional factors
        "horse_profile": "med-high",  # Age, sex - peak age 4-5yo
        "odds_fluctuations": "low",  # Historical odds movement patterns
        "stewards_comments": "med",  # Stewards notes on incidents
    }

    WEIGHT_LABELS = {
        "recent_form": "Recent Form",
        "class": "Class",
        "track_conditions": "Track Conditions",
        "distance_fit": "Distance Fit",
        "barrier_draw": "Barrier Draw",
        "jockey": "Jockey",
        "trainer": "Trainer",
        "speed_map_race_shape": "Speed Map / Race Shape",
        "weight": "Weight Carried",
        "sectionals": "Sectionals",
        "barrier_track_bias": "Barrier / Track Bias",
        "head_to_head": "Head-to-Head",
        "first_second_up": "First / Second-Up",
        "course": "Course Form",
        "gear": "Gear Changes",
        "market": "Market Movements",
        "trials": "Trials",
        "pedigree": "Pedigree / Breeding",
        "money": "Money / Stable Support",
        "tipsters_analysis": "Tipsters & Analysis",
        # Pace analysis insights
        "pf_map_factor": "Pace Advantage (Map Factor)",
        "pf_speed_rank": "Early Speed Rank",
        "pf_settle_position": "Settle Position",
        "pf_jockey_factor": "Jockey Factor",
        # Additional factors
        "horse_profile": "Horse Profile (Age/Sex)",
        "odds_fluctuations": "Odds Fluctuations",
        "stewards_comments": "Stewards Comments",
    }

    WEIGHT_OPTIONS = ["low", "low-med", "med", "med-high", "high"]

    @property
    def weights(self) -> dict[str, str]:
        """Get weights as dictionary."""
        if self.weights_json:
            stored = json.loads(self.weights_json)
            # Merge with defaults for any missing keys
            return {**self.DEFAULT_WEIGHTS, **stored}
        return self.DEFAULT_WEIGHTS.copy()

    @weights.setter
    def weights(self, value: dict[str, str]):
        """Set weights from dictionary."""
        self.weights_json = json.dumps(value)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "weights": self.weights,
            "weight_labels": self.WEIGHT_LABELS,
            "weight_options": self.WEIGHT_OPTIONS,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def format_for_prompt(self) -> str:
        """Format weights for inclusion in AI prompt."""
        lines = []
        for key, label in self.WEIGHT_LABELS.items():
            weight = self.weights.get(key, "med")
            lines.append(f"• {label}: {weight.title()}")
        return "\n".join(lines)


class AppSettings(Base):
    """General application settings."""

    __tablename__ = "app_settings"

    key = Column(String, primary_key=True)
    value = Column(Text)
    description = Column(String)
    updated_at = Column(DateTime, default=melb_now_naive, onupdate=melb_now_naive)

    # Default settings
    DEFAULTS = {
        "unit_value": {
            "value": "1",
            "description": "Value of 1U in dollars",
        },
        "persona_swearing": {
            "value": "true",
            "description": "Allow Punty to use colorful language",
        },
        "use_emojis": {
            "value": "false",
            "description": "Use emojis in content (PUNTY_MASTER says no)",
        },
        "openai_api_key": {
            "value": "",
            "description": "OpenAI API key",
        },
        "twitter_api_key": {
            "value": "",
            "description": "Twitter/X API key (Consumer Key)",
        },
        "twitter_api_secret": {
            "value": "",
            "description": "Twitter/X API secret (Consumer Secret)",
        },
        "twitter_access_token": {
            "value": "",
            "description": "Twitter/X access token",
        },
        "twitter_access_secret": {
            "value": "",
            "description": "Twitter/X access secret",
        },
        "resend_api_key": {
            "value": "",
            "description": "Resend API key for email notifications",
        },
        "email_from": {
            "value": "PuntyAI <noreply@punty.ai>",
            "description": "From address for email notifications",
        },
        "notification_email": {
            "value": "",
            "description": "Email address for scheduler notifications",
        },
        "enable_race_previews": {
            "value": "false",
            "description": "Enable individual race preview generation (future feature)",
        },
        "enable_early_mail": {
            "value": "true",
            "description": "Enable early mail generation",
        },
        "enable_meeting_wrapup": {
            "value": "true",
            "description": "Enable end of meet review generation",
        },
        "enable_results": {
            "value": "true",
            "description": "Enable per-race results commentary generation",
        },
        "telegram_bot_token": {
            "value": "",
            "description": "Telegram Bot API token (from @BotFather)",
        },
        "telegram_owner_id": {
            "value": "",
            "description": "Telegram user ID authorized to use the bot",
        },
        "anthropic_api_key": {
            "value": "",
            "description": "Anthropic API key for Claude",
        },
    }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SettingsAudit(Base):
    """Audit trail for settings changes."""

    __tablename__ = "settings_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    changed_by = Column(String(200), nullable=True)
    action = Column(String(20), nullable=False)  # created, updated, deleted
    changed_at = Column(DateTime, default=melb_now_naive, nullable=False)

    # Sensitive keys whose values should be masked in the audit log
    SENSITIVE_KEYS = {
        "openai_api_key", "twitter_api_key", "twitter_api_secret",
        "twitter_access_token", "twitter_access_secret",
        "resend_api_key", "smtp_password",
        "telegram_bot_token", "anthropic_api_key",
    }

    @staticmethod
    def mask_value(key: str, value: str | None) -> str | None:
        """Mask sensitive values, showing only first 4 and last 4 chars."""
        if value is None or not value:
            return value
        if key not in SettingsAudit.SENSITIVE_KEYS:
            return value
        if len(value) <= 8:
            return "****"
        return f"{value[:4]}...{value[-4:]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "old_value": self.mask_value(self.key, self.old_value),
            "new_value": self.mask_value(self.key, self.new_value),
            "changed_by": self.changed_by,
            "action": self.action,
            "changed_at": self.changed_at.isoformat() if self.changed_at else None,
        }


async def get_api_key(db: AsyncSession, key: str, fallback: str = "") -> str:
    """Read an API key from AppSettings DB, falling back to a default value."""
    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()
    if setting and setting.value:
        return setting.value
    return fallback
