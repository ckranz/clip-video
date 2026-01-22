"""Configuration loading and management for clip-video.

Handles loading brand and project configurations from JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PortraitSettings(BaseModel):
    """Portrait conversion settings for a brand."""

    # Horizontal crop offset (0.0=left, 0.5=center, 1.0=right)
    crop_x_offset: float = 0.5
    # For pixel-based offset on specific source resolution
    crop_x_pixels: int | None = None  # e.g., 13 for 13px from left
    crop_source_width: int | None = None  # e.g., 1280 - reference width for pixel calc


class LogoSettings(BaseModel):
    """Logo overlay settings for a brand."""

    enabled: bool = False
    # Path to logo file (relative to brand directory)
    image_path: str = "logo/logo.png"
    # Position: "top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right"
    position: str = "top-center"
    # Size as percentage of video height (0.0-1.0)
    height_percent: float = 0.15
    # Opacity (0.0-1.0, 1.0 = fully opaque)
    opacity: float = 1.0
    # Margin from edge in pixels
    margin: int = 20


class SocialCopyStyle(BaseModel):
    """Style settings for social media copy generation.

    Configures how titles, descriptions, and captions are written
    for clips when promoting content on social media.
    """

    # Whether to generate social copy alongside clips
    enabled: bool = True
    # Language/locale for spelling (british, american)
    locale: str = "british"
    # Tone: informative, casual, enthusiastic, professional
    tone: str = "informative"
    # Voice description - free text describing the brand's voice
    voice_description: str = ""
    # Phrases/patterns to avoid (AI tells, cliches, etc.)
    avoid_phrases: list[str] = Field(default_factory=lambda: [
        "game-changer",
        "crushing it",
        "let that sink in",
        "here's the thing",
        "but here's the kicker",
        "I'll be honest",
        "hot take",
        "unpopular opinion",
        "this is huge",
        "mind = blown",
    ])
    # Preferred phrases or patterns to include
    preferred_phrases: list[str] = Field(default_factory=list)
    # Whether to include hashtags
    include_hashtags: bool = True
    # Default hashtags for the brand
    default_hashtags: list[str] = Field(default_factory=list)
    # Maximum length for hook/title (characters)
    max_hook_length: int = 100
    # Maximum length for description (characters)
    max_description_length: int = 280
    # Custom system prompt additions (appended to base prompt)
    custom_prompt: str = ""


class BrandConfig(BaseModel):
    """Configuration for a brand/organization."""

    name: str
    description: str = ""
    # Default crop region for portrait conversion (center of frame) - DEPRECATED
    crop_region: dict[str, int] = Field(
        default_factory=lambda: {"x": 420, "y": 0, "width": 1080, "height": 1920}
    )
    # Portrait conversion settings
    portrait: PortraitSettings = Field(default_factory=PortraitSettings)
    # Logo overlay settings
    logo: LogoSettings = Field(default_factory=LogoSettings)
    # Social copy style settings
    social_copy: SocialCopyStyle = Field(default_factory=SocialCopyStyle)
    # Caption styling
    caption_font: str = "Arial"
    caption_size: int = 48
    caption_color: str = "#FFFFFF"
    caption_bg_color: str = "#000000"
    caption_bg_opacity: float = 0.7
    # Emoji/logo triggers (word -> emoji or logo path)
    emoji_triggers: dict[str, str] = Field(default_factory=dict)
    logo_triggers: dict[str, str] = Field(default_factory=dict)
    # Vocabulary for transcription correction
    vocabulary: dict[str, list[str]] = Field(default_factory=dict)
    # API provider preferences
    transcription_provider: str = "whisper_local"  # or "whisper_api"
    whisper_model: str = "medium"  # tiny, base, small, medium, large, large-v2, large-v3
    llm_provider: str = "claude"  # "claude", "openai", or "ollama"
    llm_model: str | None = None  # None = use provider default (e.g., claude-sonnet-4-5, gpt-4.1, llama3.2)

    def get_crop_x_offset(self, source_width: int) -> float:
        """Calculate crop X offset, handling pixel-based config.

        Args:
            source_width: Width of the source video

        Returns:
            X offset as float (0.0-1.0)
        """
        # If pixel-based offset is configured
        if self.portrait.crop_x_pixels is not None:
            ref_width = self.portrait.crop_source_width or 1280
            # Scale the pixel offset if source width differs from reference
            scaled_pixels = self.portrait.crop_x_pixels * (source_width / ref_width)
            # Calculate crop width for 9:16 from source height (assuming 16:9 source)
            source_height = int(source_width * 9 / 16)
            crop_width = int(source_height * 9 / 16)
            max_x = source_width - crop_width
            if max_x > 0:
                return scaled_pixels / max_x
            return 0.5
        return self.portrait.crop_x_offset


class ProjectConfig(BaseModel):
    """Configuration for a specific project within a brand."""

    name: str
    brand_name: str
    description: str = ""
    # Project type: "lyric_match" or "highlights"
    project_type: str = "highlights"
    # Source videos for this project
    source_videos: list[str] = Field(default_factory=list)
    # For lyric match projects
    lyrics_file: str | None = None
    # Custom settings that override brand defaults
    custom_settings: dict[str, Any] = Field(default_factory=dict)


def get_brands_root() -> Path:
    """Get the root directory for all brands.

    Returns the 'brands' directory in the current working directory.
    """
    return Path.cwd() / "brands"


def get_brand_path(brand_name: str) -> Path:
    """Get the path to a brand's directory."""
    return get_brands_root() / brand_name


def load_brand_config(brand_name: str) -> BrandConfig:
    """Load brand configuration from JSON file.

    Args:
        brand_name: Name of the brand to load

    Returns:
        BrandConfig object with the brand's settings

    Raises:
        FileNotFoundError: If brand config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = get_brand_path(brand_name) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Brand config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    return BrandConfig(**data)


def save_brand_config(brand_name: str, config: BrandConfig) -> Path:
    """Save brand configuration to JSON file with atomic write.

    Args:
        brand_name: Name of the brand
        config: BrandConfig object to save

    Returns:
        Path to the saved config file
    """
    brand_path = get_brand_path(brand_name)
    brand_path.mkdir(parents=True, exist_ok=True)

    config_path = brand_path / "config.json"
    temp_path = brand_path / "config.json.tmp"

    # Atomic write: write to temp file, then rename
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)

    temp_path.replace(config_path)
    return config_path


def load_project_config(brand_name: str, project_name: str) -> ProjectConfig:
    """Load project configuration from JSON file.

    Args:
        brand_name: Name of the brand
        project_name: Name of the project

    Returns:
        ProjectConfig object with the project's settings

    Raises:
        FileNotFoundError: If project config file doesn't exist
    """
    config_path = get_brand_path(brand_name) / "projects" / project_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    return ProjectConfig(**data)


def save_project_config(brand_name: str, project_name: str, config: ProjectConfig) -> Path:
    """Save project configuration to JSON file with atomic write.

    Args:
        brand_name: Name of the brand
        project_name: Name of the project
        config: ProjectConfig object to save

    Returns:
        Path to the saved config file
    """
    project_path = get_brand_path(brand_name) / "projects" / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    config_path = project_path / "config.json"
    temp_path = project_path / "config.json.tmp"

    # Atomic write: write to temp file, then rename
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)

    temp_path.replace(config_path)
    return config_path


def brand_exists(brand_name: str) -> bool:
    """Check if a brand exists.

    Args:
        brand_name: Name of the brand to check

    Returns:
        True if brand directory and config exist
    """
    config_path = get_brand_path(brand_name) / "config.json"
    return config_path.exists()


def list_brands() -> list[str]:
    """List all available brands.

    Returns:
        List of brand names that have valid configurations
    """
    brands_root = get_brands_root()
    if not brands_root.exists():
        return []

    brands = []
    for path in brands_root.iterdir():
        if path.is_dir() and (path / "config.json").exists():
            brands.append(path.name)

    return sorted(brands)
