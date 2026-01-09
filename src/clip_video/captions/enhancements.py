"""Brand-specific caption enhancements.

Provides functionality to inject emojis and logos when trigger words
are spoken in captions, creating branded social media content.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from clip_video.captions.styles import CaptionStyle
from clip_video.ffmpeg_binary import get_ffmpeg_path


@dataclass
class EmojiTrigger:
    """A trigger word that inserts an emoji into captions.

    Attributes:
        word: Trigger word (case-insensitive match)
        emoji: Unicode emoji to insert
        position: Where to insert ("before", "after", or "replace")
        match_whole_word: Only match complete words
    """

    word: str
    emoji: str
    position: str = "after"  # "before", "after", or "replace"
    match_whole_word: bool = True

    def apply(self, text: str) -> str:
        """Apply this trigger to text.

        Args:
            text: Text to process

        Returns:
            Text with emoji applied
        """
        if self.match_whole_word:
            pattern = rf"\b({re.escape(self.word)})\b"
        else:
            pattern = rf"({re.escape(self.word)})"

        def replacer(match):
            word = match.group(1)
            if self.position == "before":
                return f"{self.emoji} {word}"
            elif self.position == "after":
                return f"{word} {self.emoji}"
            else:  # replace
                return self.emoji

        return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


@dataclass
class LogoOverlay:
    """Configuration for logo overlay on video.

    Attributes:
        image_path: Path to logo image (PNG with transparency)
        x: X position (or expression like "W-w-10")
        y: Y position (or expression like "10")
        scale: Scale factor (1.0 = original size)
        opacity: Opacity 0.0-1.0
        trigger_word: Optional word that triggers the overlay
        duration: How long to show after trigger (seconds)
    """

    image_path: Path
    x: str = "W-w-10"  # Right side with 10px margin
    y: str = "10"  # Top with 10px margin
    scale: float = 1.0
    opacity: float = 1.0
    trigger_word: str | None = None
    duration: float = 3.0  # Duration after trigger


@dataclass
class BrandEnhancements:
    """Collection of brand-specific enhancements.

    Attributes:
        name: Brand name
        emoji_triggers: List of word-to-emoji mappings
        logo_overlays: List of logo overlays
        word_replacements: Dict of word replacements (e.g., for style)
    """

    name: str
    emoji_triggers: list[EmojiTrigger] = field(default_factory=list)
    logo_overlays: list[LogoOverlay] = field(default_factory=list)
    word_replacements: dict[str, str] = field(default_factory=dict)

    def add_emoji_trigger(
        self,
        word: str,
        emoji: str,
        position: str = "after",
    ) -> EmojiTrigger:
        """Add an emoji trigger.

        Args:
            word: Trigger word
            emoji: Emoji to insert
            position: Position relative to word

        Returns:
            The created EmojiTrigger
        """
        trigger = EmojiTrigger(word=word, emoji=emoji, position=position)
        self.emoji_triggers.append(trigger)
        return trigger

    def add_logo_overlay(
        self,
        image_path: Path | str,
        x: str = "W-w-10",
        y: str = "10",
        trigger_word: str | None = None,
    ) -> LogoOverlay:
        """Add a logo overlay.

        Args:
            image_path: Path to logo image
            x: X position expression
            y: Y position expression
            trigger_word: Optional trigger word

        Returns:
            The created LogoOverlay
        """
        overlay = LogoOverlay(
            image_path=Path(image_path),
            x=x,
            y=y,
            trigger_word=trigger_word,
        )
        self.logo_overlays.append(overlay)
        return overlay

    def enhance_text(self, text: str) -> str:
        """Apply all text enhancements to caption text.

        Args:
            text: Original caption text

        Returns:
            Enhanced text with emojis and replacements
        """
        # Apply word replacements
        for original, replacement in self.word_replacements.items():
            pattern = rf"\b{re.escape(original)}\b"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Apply emoji triggers
        for trigger in self.emoji_triggers:
            text = trigger.apply(text)

        return text

    def get_triggered_logos(
        self,
        text: str,
        current_time: float,
    ) -> list[tuple[LogoOverlay, float, float]]:
        """Get logos that should be shown based on text content.

        Args:
            text: Caption text to check for triggers
            current_time: Current timestamp in video

        Returns:
            List of (overlay, start_time, end_time) tuples
        """
        triggered = []

        for overlay in self.logo_overlays:
            if overlay.trigger_word is None:
                # Always show
                triggered.append((overlay, 0.0, float("inf")))
            else:
                # Check if trigger word is in text
                pattern = rf"\b{re.escape(overlay.trigger_word)}\b"
                if re.search(pattern, text, re.IGNORECASE):
                    start = current_time
                    end = current_time + overlay.duration
                    triggered.append((overlay, start, end))

        return triggered

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "emoji_triggers": [
                {
                    "word": t.word,
                    "emoji": t.emoji,
                    "position": t.position,
                    "match_whole_word": t.match_whole_word,
                }
                for t in self.emoji_triggers
            ],
            "logo_overlays": [
                {
                    "image_path": str(o.image_path),
                    "x": o.x,
                    "y": o.y,
                    "scale": o.scale,
                    "opacity": o.opacity,
                    "trigger_word": o.trigger_word,
                    "duration": o.duration,
                }
                for o in self.logo_overlays
            ],
            "word_replacements": self.word_replacements,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BrandEnhancements":
        """Create from dictionary."""
        enhancements = cls(name=data.get("name", ""))

        for t_data in data.get("emoji_triggers", []):
            trigger = EmojiTrigger(
                word=t_data["word"],
                emoji=t_data["emoji"],
                position=t_data.get("position", "after"),
                match_whole_word=t_data.get("match_whole_word", True),
            )
            enhancements.emoji_triggers.append(trigger)

        for o_data in data.get("logo_overlays", []):
            overlay = LogoOverlay(
                image_path=Path(o_data["image_path"]),
                x=o_data.get("x", "W-w-10"),
                y=o_data.get("y", "10"),
                scale=o_data.get("scale", 1.0),
                opacity=o_data.get("opacity", 1.0),
                trigger_word=o_data.get("trigger_word"),
                duration=o_data.get("duration", 3.0),
            )
            enhancements.logo_overlays.append(overlay)

        enhancements.word_replacements = data.get("word_replacements", {})

        return enhancements


class EnhancedCaptionRenderer:
    """Caption renderer with brand enhancement support.

    Extends base caption rendering with:
    - Emoji injection based on trigger words
    - Logo overlays (static or triggered)
    - Word replacements for styling
    """

    def __init__(self, ffmpeg_path: str | None = None):
        """Initialize the renderer.

        Args:
            ffmpeg_path: Path to FFmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()

    def render_with_enhancements(
        self,
        input_path: Path,
        output_path: Path,
        caption_track: "CaptionTrack",
        enhancements: BrandEnhancements,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
    ) -> Path:
        """Render video with enhanced captions and overlays.

        Args:
            input_path: Input video path
            output_path: Output video path
            caption_track: Caption track to render
            enhancements: Brand enhancements to apply
            video_codec: Video codec for output
            audio_codec: Audio codec for output
            crf: Quality level

        Returns:
            Path to output video
        """
        from clip_video.captions.renderer import CaptionRenderer, CaptionTrack

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply text enhancements to captions
        enhanced_track = CaptionTrack(default_style=caption_track.default_style)
        for caption in caption_track.captions:
            enhanced_text = enhancements.enhance_text(caption.text)
            enhanced_track.add_caption(
                text=enhanced_text,
                start_time=caption.start_time,
                end_time=caption.end_time,
                style=caption.style,
            )

        # Build filter complex
        filter_parts = []

        # Add static logo overlays
        input_index = 1  # [0] is main video
        overlay_inputs = []

        for overlay in enhancements.logo_overlays:
            if overlay.trigger_word is None and overlay.image_path.exists():
                overlay_inputs.append(overlay)

        # For now, use basic caption rendering
        # Full overlay support would require more complex filter graph
        renderer = CaptionRenderer(self.ffmpeg_path)
        return renderer.render(
            input_path=input_path,
            output_path=output_path,
            caption_track=enhanced_track,
            method="drawtext",
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
        )

    def apply_static_logo(
        self,
        input_path: Path,
        output_path: Path,
        logo: LogoOverlay,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
    ) -> Path:
        """Apply a static logo overlay to video.

        Args:
            input_path: Input video path
            output_path: Output video path
            logo: Logo overlay configuration
            video_codec: Video codec
            audio_codec: Audio codec
            crf: Quality level

        Returns:
            Path to output video
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not logo.image_path.exists():
            raise FileNotFoundError(f"Logo file not found: {logo.image_path}")

        # Build FFmpeg command with overlay
        scale_filter = ""
        if logo.scale != 1.0:
            scale_filter = f",scale=iw*{logo.scale}:ih*{logo.scale}"

        filter_str = (
            f"[1:v]{scale_filter}format=rgba,colorchannelmixer=aa={logo.opacity}[logo];"
            f"[0:v][logo]overlay={logo.x}:{logo.y}"
        )

        cmd = [
            self.ffmpeg_path,
            "-i", str(input_path),
            "-i", str(logo.image_path),
            "-filter_complex", filter_str,
            "-c:v", video_codec,
            "-crf", str(crf),
            "-c:a", audio_codec,
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        return output_path


# Pre-built enhancement sets for common brands
def create_cncf_enhancements() -> BrandEnhancements:
    """Create enhancements for CNCF-related content."""
    enhancements = BrandEnhancements(name="CNCF")

    # Add emoji triggers for common tech terms
    enhancements.add_emoji_trigger("Kubernetes", "☸️")
    enhancements.add_emoji_trigger("K8s", "☸️")
    enhancements.add_emoji_trigger("cloud", "☁️")
    enhancements.add_emoji_trigger("container", "📦")
    enhancements.add_emoji_trigger("security", "🔒")
    enhancements.add_emoji_trigger("open source", "💚")
    enhancements.add_emoji_trigger("community", "🤝")
    enhancements.add_emoji_trigger("scale", "📈")
    enhancements.add_emoji_trigger("scaling", "📈")
    enhancements.add_emoji_trigger("deploy", "🚀")
    enhancements.add_emoji_trigger("deployment", "🚀")

    return enhancements


def create_tech_enhancements() -> BrandEnhancements:
    """Create generic tech content enhancements."""
    enhancements = BrandEnhancements(name="Tech")

    enhancements.add_emoji_trigger("important", "⚡")
    enhancements.add_emoji_trigger("tip", "💡")
    enhancements.add_emoji_trigger("warning", "⚠️")
    enhancements.add_emoji_trigger("error", "❌")
    enhancements.add_emoji_trigger("success", "✅")
    enhancements.add_emoji_trigger("code", "💻")
    enhancements.add_emoji_trigger("API", "🔌")
    enhancements.add_emoji_trigger("database", "🗄️")
    enhancements.add_emoji_trigger("performance", "⚡")
    enhancements.add_emoji_trigger("bug", "🐛")
    enhancements.add_emoji_trigger("fix", "🔧")

    return enhancements
