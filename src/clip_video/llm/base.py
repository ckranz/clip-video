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
    OLLAMA = "ollama"


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
                self.model = "claude-sonnet-4-5-20250929"
            elif self.provider == LLMProviderType.OPENAI:
                self.model = "gpt-4.1"
            elif self.provider == LLMProviderType.OLLAMA:
                self.model = "llama3.2"


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


@dataclass
class ClipValidationRequest:
    """Request for LLM to validate a clip segment.

    D002-D007: Quality criteria for LLM validation
    - sentence_boundaries: Complete sentences at start/end
    - topic_complete: Topic is complete, not cut off mid-thought
    - has_hook: Contains engaging opening (optional but valued)
    - standalone_valid: Makes sense without surrounding context
    - brand_relevant: Aligns with brand context if provided
    - transcript_aligned: Timestamps align with actual content

    Attributes:
        clip_id: Unique identifier for the clip
        transcript_segment: The transcript text for this segment
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        clip_summary: Brief summary of the clip content
        brand_context: Brand/channel context for relevance check
        quality_criteria: List of criteria to evaluate
        full_transcript_context: Optional surrounding context for validation
    """

    clip_id: str
    transcript_segment: str
    start_time: float
    end_time: float
    clip_summary: str
    brand_context: dict[str, Any] = field(default_factory=dict)
    quality_criteria: list[str] = field(default_factory=lambda: [
        "sentence_boundaries",
        "topic_complete",
        "has_hook",
        "standalone_valid",
        "brand_relevant",
        "transcript_aligned",
    ])
    full_transcript_context: str | None = None

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "transcript_segment": self.transcript_segment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "clip_summary": self.clip_summary,
            "brand_context": self.brand_context,
            "quality_criteria": self.quality_criteria,
            "full_transcript_context": self.full_transcript_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipValidationRequest":
        """Create from dictionary."""
        return cls(
            clip_id=data["clip_id"],
            transcript_segment=data["transcript_segment"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            clip_summary=data["clip_summary"],
            brand_context=data.get("brand_context", {}),
            quality_criteria=data.get("quality_criteria", [
                "sentence_boundaries",
                "topic_complete",
                "has_hook",
                "standalone_valid",
                "brand_relevant",
                "transcript_aligned",
            ]),
            full_transcript_context=data.get("full_transcript_context"),
        )


@dataclass
class ClipValidationResponse:
    """Response from LLM clip validation.

    Contains pass/fail for each quality criterion with detailed feedback.

    Attributes:
        clip_id: Identifier of the validated clip
        is_valid: Overall validation result (all required criteria pass)
        sentence_boundaries_ok: D002 - Complete sentences at start/end
        topic_complete: D003 - Topic is complete, not cut off
        has_hook: D004 - Contains engaging opening (optional)
        standalone_valid: D005 - Makes sense without context
        brand_relevant: D006 - Aligns with brand context
        transcript_aligned: D007 - Timestamps align with content
        issues: List of specific issues found
        suggestions: Suggestions for improvement or alternatives
        confidence: LLM confidence in this assessment (0-1)
        tokens_used: Tokens consumed for this validation
    """

    clip_id: str
    is_valid: bool
    sentence_boundaries_ok: bool = True
    topic_complete: bool = True
    has_hook: bool = False  # Optional but valued
    standalone_valid: bool = True
    brand_relevant: bool = True
    transcript_aligned: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.8
    tokens_used: int = 0

    @property
    def required_criteria_passed(self) -> bool:
        """Check if all required criteria pass.

        has_hook is optional, all others are required.
        """
        return (
            self.sentence_boundaries_ok
            and self.topic_complete
            and self.standalone_valid
            and self.brand_relevant
            and self.transcript_aligned
        )

    @property
    def failed_criteria(self) -> list[str]:
        """Get list of failed criteria names."""
        failed = []
        if not self.sentence_boundaries_ok:
            failed.append("sentence_boundaries")
        if not self.topic_complete:
            failed.append("topic_complete")
        if not self.standalone_valid:
            failed.append("standalone_valid")
        if not self.brand_relevant:
            failed.append("brand_relevant")
        if not self.transcript_aligned:
            failed.append("transcript_aligned")
        return failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "is_valid": self.is_valid,
            "sentence_boundaries_ok": self.sentence_boundaries_ok,
            "topic_complete": self.topic_complete,
            "has_hook": self.has_hook,
            "standalone_valid": self.standalone_valid,
            "brand_relevant": self.brand_relevant,
            "transcript_aligned": self.transcript_aligned,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "required_criteria_passed": self.required_criteria_passed,
            "failed_criteria": self.failed_criteria,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipValidationResponse":
        """Create from dictionary."""
        return cls(
            clip_id=data["clip_id"],
            is_valid=data["is_valid"],
            sentence_boundaries_ok=data.get("sentence_boundaries_ok", True),
            topic_complete=data.get("topic_complete", True),
            has_hook=data.get("has_hook", False),
            standalone_valid=data.get("standalone_valid", True),
            brand_relevant=data.get("brand_relevant", True),
            transcript_aligned=data.get("transcript_aligned", True),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            confidence=data.get("confidence", 0.8),
            tokens_used=data.get("tokens_used", 0),
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

    @abstractmethod
    def validate_clip(
        self,
        request: ClipValidationRequest,
    ) -> ClipValidationResponse:
        """Validate a clip segment against quality criteria.

        Uses LLM to evaluate whether the clip meets quality standards
        for sentence boundaries, topic completeness, standalone validity,
        brand relevance, and transcript alignment.

        Args:
            request: ClipValidationRequest with segment details and criteria

        Returns:
            ClipValidationResponse with pass/fail for each criterion
        """
        pass

    @abstractmethod
    def find_replacement_clips(
        self,
        rejected_clips: list[ClipValidationResponse],
        transcript_text: str,
        used_segments: list[tuple[float, float]],
        target_count: int = 1,
    ) -> list[HighlightSegment]:
        """Find replacement clips for rejected segments.

        Given clips that failed validation, identify alternative segments
        from the transcript that could serve as replacements, avoiding
        already-used time ranges.

        Args:
            rejected_clips: List of clips that failed validation
            transcript_text: Full transcript for finding alternatives
            used_segments: List of (start, end) tuples to avoid
            target_count: Number of replacement clips needed

        Returns:
            List of new HighlightSegment suggestions
        """
        pass
