"""Caption rendering module for burning subtitles into videos.

Provides tools for rendering captions using FFmpeg drawtext filter,
with support for configurable fonts, styles, and positioning.
"""

from clip_video.captions.styles import (
    CaptionStyle,
    CaptionPosition,
    DEFAULT_STYLE,
    YOUTUBE_SHORTS_STYLE,
    TIKTOK_STYLE,
)
from clip_video.captions.renderer import (
    CaptionRenderer,
    Caption,
    CaptionTrack,
)

__all__ = [
    "CaptionStyle",
    "CaptionPosition",
    "DEFAULT_STYLE",
    "YOUTUBE_SHORTS_STYLE",
    "TIKTOK_STYLE",
    "CaptionRenderer",
    "Caption",
    "CaptionTrack",
]
