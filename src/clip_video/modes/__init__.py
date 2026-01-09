"""Processing modes for clip-video.

Two primary modes:
- lyric_match: Build searchable word/phrase dictionaries for music video mashups
- highlights: Batch-process recordings for social media shorts
"""

from clip_video.modes.lyric_match import (
    LyricMatchProject,
    LyricMatchProcessor,
)

__all__ = [
    "LyricMatchProject",
    "LyricMatchProcessor",
]
