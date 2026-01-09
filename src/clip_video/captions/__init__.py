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
from clip_video.captions.enhancements import (
    EmojiTrigger,
    LogoOverlay,
    BrandEnhancements,
    EnhancedCaptionRenderer,
    create_cncf_enhancements,
    create_tech_enhancements,
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
    "EmojiTrigger",
    "LogoOverlay",
    "BrandEnhancements",
    "EnhancedCaptionRenderer",
    "create_cncf_enhancements",
    "create_tech_enhancements",
]
