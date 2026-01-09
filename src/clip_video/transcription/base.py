"""Base classes for transcription providers.

Defines the abstract interface that all transcription providers must implement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionWord:
    """A single word with timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0
    original_word: str | None = None  # If corrected, the original word

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }
        if self.original_word is not None:
            data["original_word"] = self.original_word
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionWord":
        """Create from dictionary."""
        return cls(
            word=data["word"],
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            original_word=data.get("original_word"),
        )


@dataclass
class TranscriptionSegment:
    """A segment of transcription with timing and words."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    words: list[TranscriptionWord] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionSegment":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            start=data["start"],
            end=data["end"],
            words=[TranscriptionWord.from_dict(w) for w in data.get("words", [])],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    video_path: str
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0
    provider: str = ""
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    cost_usd: float | None = None
    vocabulary_corrections: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "duration": self.duration,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "cost_usd": self.cost_usd,
            "vocabulary_corrections": self.vocabulary_corrections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        """Create from dictionary."""
        return cls(
            video_path=data["video_path"],
            text=data["text"],
            segments=[TranscriptionSegment.from_dict(s) for s in data.get("segments", [])],
            language=data.get("language", "en"),
            duration=data.get("duration", 0.0),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            cost_usd=data.get("cost_usd"),
            vocabulary_corrections=data.get("vocabulary_corrections", 0),
        )

    def get_all_words(self) -> list[TranscriptionWord]:
        """Get all words from all segments."""
        words = []
        for segment in self.segments:
            words.extend(segment.words)
        return words

    def save(self, path: Path | str) -> None:
        """Save transcription to JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "TranscriptionResult":
        """Load transcription from JSON file.

        Args:
            path: Path to load from

        Returns:
            TranscriptionResult instance
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers.

    All transcription providers (OpenAI API, local whisper, etc.)
    must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "en",
        prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe an audio/video file.

        Args:
            audio_path: Path to audio or video file
            language: Language code (e.g., "en", "es")
            prompt: Optional prompt for conditioning

        Returns:
            TranscriptionResult with transcribed content
        """
        pass

    @abstractmethod
    def estimate_cost(self, duration_seconds: float) -> float | None:
        """Estimate cost for transcribing audio of given duration.

        Args:
            duration_seconds: Duration of audio in seconds

        Returns:
            Estimated cost in USD, or None if not applicable (local)
        """
        pass

    def is_available(self) -> bool:
        """Check if the provider is available and configured.

        Returns:
            True if provider can be used
        """
        return True
