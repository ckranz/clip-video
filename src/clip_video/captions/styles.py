"""Caption styling configuration.

Defines styles for caption rendering including fonts, colors,
positioning, and text formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class CaptionPosition(str, Enum):
    """Vertical position for captions."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    LOWER_THIRD = "lower_third"  # Lower 1/3 of screen


@dataclass
class CaptionStyle:
    """Configuration for caption appearance.

    Attributes:
        font_family: Font family name (must be available on system)
        font_size: Font size in pixels
        font_color: Font color as hex (e.g., "FFFFFF" for white)
        font_opacity: Font opacity 0.0-1.0
        background_color: Background box color as hex
        background_opacity: Background opacity 0.0-1.0
        border_color: Text border/outline color as hex
        border_width: Text border width in pixels
        position: Vertical position on screen
        margin_x: Horizontal margin in pixels
        margin_y: Vertical margin in pixels
        line_spacing: Space between lines in pixels
        max_width_percent: Maximum width as percentage of video width
        shadow_color: Drop shadow color as hex
        shadow_offset: Drop shadow offset (x, y) in pixels
        word_wrap: Enable automatic word wrapping
        uppercase: Convert text to uppercase
        bold: Use bold font weight
    """

    font_family: str = "Arial"
    font_size: int = 48
    font_color: str = "FFFFFF"
    font_opacity: float = 1.0
    background_color: str = "000000"
    background_opacity: float = 0.5
    border_color: str = "000000"
    border_width: int = 2
    position: CaptionPosition = CaptionPosition.LOWER_THIRD
    margin_x: int = 40
    margin_y: int = 60
    line_spacing: int = 10
    max_width_percent: float = 0.9
    shadow_color: str = "000000"
    shadow_offset: Tuple[int, int] = (2, 2)
    word_wrap: bool = True
    uppercase: bool = False
    bold: bool = True

    def get_y_position(self, video_height: int) -> int:
        """Calculate Y position for captions.

        Args:
            video_height: Height of the video in pixels

        Returns:
            Y coordinate for caption placement
        """
        if self.position == CaptionPosition.TOP:
            return self.margin_y
        elif self.position == CaptionPosition.CENTER:
            return video_height // 2
        elif self.position == CaptionPosition.BOTTOM:
            return video_height - self.margin_y - self.font_size
        elif self.position == CaptionPosition.LOWER_THIRD:
            # Position in lower third of screen
            return int(video_height * 0.75)
        else:
            return video_height - self.margin_y - self.font_size

    def get_ffmpeg_color(self, hex_color: str, opacity: float = 1.0) -> str:
        """Convert hex color to FFmpeg format with opacity.

        Args:
            hex_color: Color as hex string (e.g., "FFFFFF")
            opacity: Opacity value 0.0-1.0

        Returns:
            FFmpeg color string
        """
        # FFmpeg uses RGBA format
        alpha = int(opacity * 255)
        return f"0x{hex_color}{alpha:02X}"

    def get_font_string(self) -> str:
        """Get font specification string.

        Returns:
            Font string for FFmpeg
        """
        return self.font_family

    def to_drawtext_params(self, video_width: int, video_height: int) -> dict:
        """Convert style to FFmpeg drawtext parameters.

        Args:
            video_width: Width of the video
            video_height: Height of the video

        Returns:
            Dict of drawtext filter parameters
        """
        y_pos = self.get_y_position(video_height)
        max_width = int(video_width * self.max_width_percent)

        params = {
            "fontsize": self.font_size,
            "fontcolor": self.get_ffmpeg_color(self.font_color, self.font_opacity),
            "x": "(w-text_w)/2",  # Center horizontally
            "y": y_pos,
            "borderw": self.border_width,
            "bordercolor": self.get_ffmpeg_color(self.border_color, 1.0),
        }

        # Add shadow if specified
        if self.shadow_offset != (0, 0):
            params["shadowcolor"] = self.get_ffmpeg_color(self.shadow_color, 0.5)
            params["shadowx"] = self.shadow_offset[0]
            params["shadowy"] = self.shadow_offset[1]

        # Add background box if opacity > 0
        if self.background_opacity > 0:
            params["box"] = 1
            params["boxcolor"] = self.get_ffmpeg_color(
                self.background_color, self.background_opacity
            )
            params["boxborderw"] = 10

        return params


# Pre-built styles for common platforms
DEFAULT_STYLE = CaptionStyle()

YOUTUBE_SHORTS_STYLE = CaptionStyle(
    font_family="Arial",
    font_size=56,
    font_color="FFFFFF",
    background_color="000000",
    background_opacity=0.6,
    border_width=2,
    border_color="000000",
    position=CaptionPosition.LOWER_THIRD,
    margin_y=100,
    bold=True,
)

TIKTOK_STYLE = CaptionStyle(
    font_family="Arial",
    font_size=64,
    font_color="FFFFFF",
    background_color="000000",
    background_opacity=0.0,  # No background box
    border_width=3,
    border_color="000000",
    shadow_offset=(3, 3),
    position=CaptionPosition.CENTER,
    uppercase=True,
    bold=True,
)

LINKEDIN_STYLE = CaptionStyle(
    font_family="Arial",
    font_size=42,
    font_color="FFFFFF",
    background_color="0A66C2",  # LinkedIn blue
    background_opacity=0.8,
    border_width=0,
    position=CaptionPosition.BOTTOM,
    margin_y=40,
)
