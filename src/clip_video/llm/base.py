"""Base classes and types for LLM integration.

Defines the abstract interface for LLM providers and data structures
for highlight analysis results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    CLAUDE = "claude"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    Attributes:
        provider: Which LLM provider to use
        api_key: API key for the provider (or use env variable)
        model: Model name/ID to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-1)
        timeout: Request timeout in seconds
    """

    provider: LLMProviderType = LLMProviderType.CLAUDE
    api_key: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 120

    def __post_init__(self):
        """Set default models based on provider."""
        if self.model is None:
            if self.provider == LLMProviderType.CLAUDE:
                self.model = "claude-sonnet-4-20250514"
            elif self.provider == LLMProviderType.OPENAI:
                self.model = "gpt-4o"


@dataclass
class HighlightSegment:
    """A single highlight segment identified by LLM analysis.

    Attributes:
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        summary: Brief summary of what happens in this segment
        hook_text: Suggested hook/caption for social media
        reason: Why this segment was selected as a highlight
        topics: Key topics/themes in this segment
        quality_score: Estimated quality score (0-1)
    """

    start_time: float
    end_time: float
    summary: str
    hook_text: str
    reason: str
    topics: list[str] = field(default_factory=list)
    quality_score: float = 0.8

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "hook_text": self.hook_text,
            "reason": self.reason,
            "topics": self.topics,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HighlightSegment":
        """Create from dictionary."""
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            summary=data["summary"],
            hook_text=data["hook_text"],
            reason=data["reason"],
            topics=data.get("topics", []),
            quality_score=data.get("quality_score", 0.8),
        )


@dataclass
class HighlightAnalysis:
    """Complete analysis result from LLM.

    Attributes:
        video_id: Identifier of the analyzed video
        segments: List of identified highlight segments
        session_summary: Overall summary of the session content
        main_topics: Key topics covered in the session
        recommended_count: Suggested number of clips to extract
        model_used: Which model performed the analysis
        tokens_used: Number of tokens consumed
        cost_estimate: Estimated cost in USD
    """

    video_id: str
    segments: list[HighlightSegment]
    session_summary: str = ""
    main_topics: list[str] = field(default_factory=list)
    recommended_count: int = 3
    model_used: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0

    @property
    def total_duration(self) -> float:
        """Total duration of all highlight segments."""
        return sum(seg.duration for seg in self.segments)

    def top_segments(self, n: int = 3) -> list[HighlightSegment]:
        """Get top N segments by quality score.

        Args:
            n: Number of segments to return

        Returns:
            Top N segments sorted by quality
        """
        return sorted(
            self.segments,
            key=lambda s: s.quality_score,
            reverse=True,
        )[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "segments": [s.to_dict() for s in self.segments],
            "session_summary": self.session_summary,
            "main_topics": self.main_topics,
            "recommended_count": self.recommended_count,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HighlightAnalysis":
        """Create from dictionary."""
        return cls(
            video_id=data["video_id"],
            segments=[HighlightSegment.from_dict(s) for s in data.get("segments", [])],
            session_summary=data.get("session_summary", ""),
            main_topics=data.get("main_topics", []),
            recommended_count=data.get("recommended_count", 3),
            model_used=data.get("model_used", ""),
            tokens_used=data.get("tokens_used", 0),
            cost_estimate=data.get("cost_estimate", 0.0),
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides interface for transcript analysis and highlight detection.
    Implementations handle provider-specific API calls.
    """

    def __init__(self, config: LLMConfig):
        """Initialize the provider.

        Args:
            config: LLM configuration
        """
        self.config = config

    @abstractmethod
    def analyze_transcript(
        self,
        transcript_text: str,
        session_description: str | None = None,
        target_clips: int = 3,
    ) -> HighlightAnalysis:
        """Analyze a transcript to identify highlight segments.

        Args:
            transcript_text: Full transcript text with timestamps
            session_description: Optional description of the session content
            target_clips: Target number of highlight clips to identify

        Returns:
            HighlightAnalysis with identified segments
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        transcript_text: str,
    ) -> float:
        """Estimate the cost of analyzing a transcript.

        Args:
            transcript_text: Transcript to estimate for

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (API key set, etc).

        Returns:
            True if provider can be used
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
