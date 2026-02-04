"""Settings models for configurable analysis weights and preferences."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, Text, select
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
        "market": "med",
        "trials": "med-high",
        "pedigree": "low-med",
        "money": "low-med",
        "tipsters_analysis": "high",
        # Punting Form insights (pace/speed metrics)
        "pf_map_factor": "high",  # Pace advantage/disadvantage factor
        "pf_speed_rank": "high",  # Early speed rating (1=fastest)
        "pf_settle_position": "high",  # Historical settling position
        "pf_jockey_factor": "med-high",  # Jockey effectiveness metric
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
        # Punting Form insights
        "pf_map_factor": "PF Pace Advantage (Map Factor)",
        "pf_speed_rank": "PF Early Speed Rank",
        "pf_settle_position": "PF Settle Position",
        "pf_jockey_factor": "PF Jockey Factor",
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
        "whatsapp_invite_link": {
            "value": "https://chat.whatsapp.com/GfYvzcQ4f4L6o0XMw0lgx1",
            "description": "WhatsApp group invite link for Early Mail",
        },
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
        "favorites_threshold": {
            "value": "2.50",
            "description": "Odds threshold below which a selection is considered a favorite",
        },
        "auto_widen_favorites": {
            "value": "false",
            "description": "Automatically ask AI to reconsider when too many favorites selected",
        },
        "max_favorites_per_card": {
            "value": "3",
            "description": "Maximum favorites allowed before triggering widen request",
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
        "whatsapp_api_token": {
            "value": "",
            "description": "WhatsApp Business API token",
        },
        "whatsapp_phone_number_id": {
            "value": "",
            "description": "WhatsApp phone number ID",
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
    }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


async def get_api_key(db: AsyncSession, key: str, fallback: str = "") -> str:
    """Read an API key from AppSettings DB, falling back to a default value."""
    result = await db.execute(select(AppSettings).where(AppSettings.key == key))
    setting = result.scalar_one_or_none()
    if setting and setting.value:
        return setting.value
    return fallback
