"""Video processing module.

Provides tools for video format conversion, cropping, and optimization
for social media platforms.
"""

from clip_video.video.portrait import (
    PortraitConfig,
    PortraitConverter,
    AspectRatio,
    calculate_crop_region,
)

__all__ = [
    "PortraitConfig",
    "PortraitConverter",
    "AspectRatio",
    "calculate_crop_region",
]
