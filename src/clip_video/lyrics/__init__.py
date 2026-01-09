"""Lyrics parsing and phrase extraction module.

Provides tools for parsing lyrics files and extracting words/phrases
for building searchable dictionaries from video transcripts.
"""

from clip_video.lyrics.parser import LyricsParser, LyricsLine, ParsedLyrics
from clip_video.lyrics.phrases import (
    PhraseExtractor,
    ExtractionTarget,
    ExtractionList,
)

__all__ = [
    "LyricsParser",
    "LyricsLine",
    "ParsedLyrics",
    "PhraseExtractor",
    "ExtractionTarget",
    "ExtractionList",
]
