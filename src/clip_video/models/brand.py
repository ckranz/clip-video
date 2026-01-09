"""Brand configuration model.

A Brand represents an organization or content creator with specific styling preferences.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CropRegion(BaseModel):
    """Defines a rectangular region for cropping video to portrait mode."""

    x: int = 420
    y: int = 0
    width: int = 1080
    height: int = 1920

    def to_ffmpeg_filter(self) -> str:
        """Convert to ffmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


class CaptionStyle(BaseModel):
    """Styling configuration for video captions."""

    font: str = "Arial"
    size: int = 48
    color: str = "#FFFFFF"
    bg_color: str = "#000000"
    bg_opacity: float = 0.7
    position: str = "bottom"  # "top", "center", "bottom"
    margin_bottom: int = 100  # pixels from bottom when position is "bottom"


class Brand(BaseModel):
    """Configuration for a brand/organization.

    A brand contains styling preferences, vocabulary for transcription,
    and triggers for emoji/logo overlays.
    """

    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Default crop region for portrait conversion
    crop_region: CropRegion = Field(default_factory=CropRegion)

    # Caption styling
    caption_style: CaptionStyle = Field(default_factory=CaptionStyle)

    # Emoji triggers: word/phrase -> emoji character
    emoji_triggers: dict[str, str] = Field(default_factory=dict)

    # Logo triggers: word/phrase -> path to logo image
    logo_triggers: dict[str, str] = Field(default_factory=dict)

    # Vocabulary for transcription correction
    # Format: canonical_word -> [alternative_spellings]
    vocabulary: dict[str, list[str]] = Field(default_factory=dict)

    # API provider preferences
    transcription_provider: str = "whisper_api"  # "whisper_api" or "whisper_local"
    llm_provider: str = "claude"  # "claude" or "openai"

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now()

    def add_vocabulary(self, word: str, alternatives: list[str]) -> None:
        """Add vocabulary correction mapping.

        Args:
            word: The canonical spelling
            alternatives: List of alternative spellings to correct
        """
        self.vocabulary[word] = alternatives
        self.update_timestamp()

    def add_emoji_trigger(self, trigger: str, emoji: str) -> None:
        """Add an emoji trigger.

        Args:
            trigger: Word or phrase that triggers the emoji
            emoji: Emoji character to display
        """
        self.emoji_triggers[trigger.lower()] = emoji
        self.update_timestamp()

    def add_logo_trigger(self, trigger: str, logo_path: str) -> None:
        """Add a logo trigger.

        Args:
            trigger: Word or phrase that triggers the logo
            logo_path: Path to the logo image file
        """
        self.logo_triggers[trigger.lower()] = logo_path
        self.update_timestamp()

    def get_correction(self, word: str) -> str | None:
        """Get the canonical spelling for a word if it needs correction.

        Args:
            word: The word to check

        Returns:
            The canonical spelling if correction needed, None otherwise
        """
        word_lower = word.lower()
        for canonical, alternatives in self.vocabulary.items():
            if word_lower in [alt.lower() for alt in alternatives]:
                return canonical
        return None
