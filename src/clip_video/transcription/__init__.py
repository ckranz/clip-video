"""Transcription module for clip-video.

Provides Whisper-based transcription with support for both
OpenAI API and local whisper implementations.
"""

from clip_video.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from clip_video.transcription.progress import TranscriptionProgress
from clip_video.transcription.whisper_api import WhisperAPIProvider
from clip_video.transcription.whisper_local import WhisperLocalProvider

__all__ = [
    "TranscriptionProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "TranscriptionProgress",
    "WhisperAPIProvider",
    "WhisperLocalProvider",
]
