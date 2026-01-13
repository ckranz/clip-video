"""Cost tracking and reporting for clip-video.

Tracks API costs throughout processing and generates summaries:
- Transcription costs (Whisper API)
- LLM costs (Claude, OpenAI)
- Running totals during processing
- Final cost reports with breakdowns
- Time and cost estimates for future batches
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from clip_video.storage import atomic_write_json, read_json


class CostCategory(str, Enum):
    """Categories of costs."""

    TRANSCRIPTION = "transcription"
    LLM_ANALYSIS = "llm_analysis"
    OTHER = "other"


@dataclass
class CostEntry:
    """A single cost entry.

    Attributes:
        category: Cost category
        service: Service name (e.g., "whisper", "claude", "openai")
        amount: Cost in USD
        description: Description of what the cost is for
        tokens_used: Number of tokens (for LLM costs)
        duration_seconds: Duration processed (for transcription)
        timestamp: When the cost was incurred
        metadata: Additional metadata
    """

    category: CostCategory
    service: str
    amount: float
    description: str = ""
    tokens_used: int = 0
    duration_seconds: float = 0.0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "service": self.service,
            "amount": self.amount,
            "description": self.description,
            "tokens_used": self.tokens_used,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostEntry":
        """Create from dictionary."""
        return cls(
            category=CostCategory(data["category"]),
            service=data["service"],
            amount=data["amount"],
            description=data.get("description", ""),
            tokens_used=data.get("tokens_used", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CostSummary:
    """Summary of costs for a category or service.

    Attributes:
        category: Category or "total"
        service: Service or "all"
        total_amount: Total cost in USD
        entry_count: Number of entries
        total_tokens: Total tokens used
        total_duration: Total duration processed
    """

    category: str
    service: str
    total_amount: float
    entry_count: int
    total_tokens: int = 0
    total_duration: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "service": self.service,
            "total_amount": round(self.total_amount, 6),
            "entry_count": self.entry_count,
            "total_tokens": self.total_tokens,
            "total_duration": round(self.total_duration, 2),
        }


class CostTracker:
    """Tracks costs throughout processing.

    Provides real-time cost tracking with:
    - Running totals
    - Category and service breakdowns
    - Cost estimation
    - Persistence
    - Cost ceiling enforcement (for agentic workflows)

    Example:
        tracker = CostTracker(ceiling_gbp=5.0)
        tracker.add_transcription_cost(
            service="whisper",
            amount=0.50,
            duration_seconds=300,
        )
        tracker.add_llm_cost(
            service="claude",
            amount=0.02,
            tokens_used=1500,
        )
        print(tracker.get_total())
        print(tracker.ceiling_reached)  # Check if limit hit
    """

    # GBP to USD exchange rate (configurable)
    USD_TO_GBP = 0.79

    def __init__(
        self,
        cost_callback: Callable[[float, str], None] | None = None,
        ceiling_gbp: float | None = None,
    ):
        """Initialize the tracker.

        Args:
            cost_callback: Optional callback when costs are added
                Signature: (amount, description)
            ceiling_gbp: Optional cost ceiling in GBP for agentic workflows
        """
        self.entries: list[CostEntry] = []
        self.cost_callback = cost_callback
        self._started_at = datetime.now().isoformat()
        self._ceiling_gbp = ceiling_gbp

    def add_cost(self, entry: CostEntry) -> None:
        """Add a cost entry.

        Args:
            entry: Cost entry to add
        """
        self.entries.append(entry)

        if self.cost_callback:
            self.cost_callback(entry.amount, entry.description)

    def add_transcription_cost(
        self,
        service: str,
        amount: float,
        duration_seconds: float,
        description: str = "",
        **metadata: dict,
    ) -> CostEntry:
        """Add a transcription cost entry.

        Args:
            service: Transcription service name
            amount: Cost in USD
            duration_seconds: Audio duration transcribed
            description: Optional description
            **metadata: Additional metadata

        Returns:
            Created CostEntry
        """
        entry = CostEntry(
            category=CostCategory.TRANSCRIPTION,
            service=service,
            amount=amount,
            description=description or f"Transcription ({duration_seconds:.1f}s audio)",
            duration_seconds=duration_seconds,
            metadata=metadata,
        )
        self.add_cost(entry)
        return entry

    def add_llm_cost(
        self,
        service: str,
        amount: float,
        tokens_used: int,
        description: str = "",
        **metadata: dict,
    ) -> CostEntry:
        """Add an LLM cost entry.

        Args:
            service: LLM service name
            amount: Cost in USD
            tokens_used: Number of tokens used
            description: Optional description
            **metadata: Additional metadata

        Returns:
            Created CostEntry
        """
        entry = CostEntry(
            category=CostCategory.LLM_ANALYSIS,
            service=service,
            amount=amount,
            description=description or f"LLM analysis ({tokens_used} tokens)",
            tokens_used=tokens_used,
            metadata=metadata,
        )
        self.add_cost(entry)
        return entry

    def add_other_cost(
        self,
        service: str,
        amount: float,
        description: str = "",
        **metadata: dict,
    ) -> CostEntry:
        """Add a miscellaneous cost entry.

        Args:
            service: Service name
            amount: Cost in USD
            description: Description
            **metadata: Additional metadata

        Returns:
            Created CostEntry
        """
        entry = CostEntry(
            category=CostCategory.OTHER,
            service=service,
            amount=amount,
            description=description,
            metadata=metadata,
        )
        self.add_cost(entry)
        return entry

    def get_total(self) -> float:
        """Get total cost across all entries.

        Returns:
            Total cost in USD
        """
        return sum(e.amount for e in self.entries)

    def get_total_by_category(self, category: CostCategory) -> float:
        """Get total cost for a category.

        Args:
            category: Cost category

        Returns:
            Total cost in USD
        """
        return sum(e.amount for e in self.entries if e.category == category)

    def get_total_by_service(self, service: str) -> float:
        """Get total cost for a service.

        Args:
            service: Service name

        Returns:
            Total cost in USD
        """
        return sum(e.amount for e in self.entries if e.service == service)

    def get_total_gbp(self) -> float:
        """Get total cost in GBP.

        Returns:
            Total cost in GBP
        """
        return self.get_total() * self.USD_TO_GBP

    @property
    def ceiling_gbp(self) -> float | None:
        """Get the cost ceiling in GBP.

        Returns:
            Cost ceiling in GBP, or None if no ceiling set
        """
        return self._ceiling_gbp

    @ceiling_gbp.setter
    def ceiling_gbp(self, value: float | None) -> None:
        """Set the cost ceiling in GBP.

        Args:
            value: Cost ceiling in GBP, or None to disable
        """
        self._ceiling_gbp = value

    @property
    def remaining_gbp(self) -> float:
        """Get remaining budget in GBP.

        Returns:
            Remaining budget, or infinity if no ceiling set
        """
        if self._ceiling_gbp is None:
            return float("inf")
        return max(0, self._ceiling_gbp - self.get_total_gbp())

    @property
    def ceiling_reached(self) -> bool:
        """Check if cost ceiling has been reached.

        Returns:
            True if ceiling reached or exceeded, False otherwise
        """
        if self._ceiling_gbp is None:
            return False
        return self.get_total_gbp() >= self._ceiling_gbp

    def can_afford(self, estimated_cost_usd: float) -> bool:
        """Check if an estimated cost can be afforded.

        Args:
            estimated_cost_usd: Estimated cost of next operation in USD

        Returns:
            True if cost is within remaining budget
        """
        if self._ceiling_gbp is None:
            return True
        estimated_gbp = estimated_cost_usd * self.USD_TO_GBP
        return (self.get_total_gbp() + estimated_gbp) <= self._ceiling_gbp

    def get_summary_by_category(self) -> dict[str, CostSummary]:
        """Get cost summaries grouped by category.

        Returns:
            Dict mapping category to CostSummary
        """
        summaries = {}

        for category in CostCategory:
            entries = [e for e in self.entries if e.category == category]
            if entries:
                summaries[category.value] = CostSummary(
                    category=category.value,
                    service="all",
                    total_amount=sum(e.amount for e in entries),
                    entry_count=len(entries),
                    total_tokens=sum(e.tokens_used for e in entries),
                    total_duration=sum(e.duration_seconds for e in entries),
                )

        return summaries

    def get_summary_by_service(self) -> dict[str, CostSummary]:
        """Get cost summaries grouped by service.

        Returns:
            Dict mapping service to CostSummary
        """
        summaries = {}
        services = set(e.service for e in self.entries)

        for service in services:
            entries = [e for e in self.entries if e.service == service]
            summaries[service] = CostSummary(
                category="all",
                service=service,
                total_amount=sum(e.amount for e in entries),
                entry_count=len(entries),
                total_tokens=sum(e.tokens_used for e in entries),
                total_duration=sum(e.duration_seconds for e in entries),
            )

        return summaries

    def get_full_report(self) -> dict:
        """Generate a full cost report.

        Returns:
            Complete cost report dictionary
        """
        by_category = self.get_summary_by_category()
        by_service = self.get_summary_by_service()

        report = {
            "total_cost_usd": round(self.get_total(), 6),
            "total_cost_gbp": round(self.get_total_gbp(), 6),
            "entry_count": len(self.entries),
            "started_at": self._started_at,
            "generated_at": datetime.now().isoformat(),
            "by_category": {k: v.to_dict() for k, v in by_category.items()},
            "by_service": {k: v.to_dict() for k, v in by_service.items()},
            "entries": [e.to_dict() for e in self.entries],
        }

        # Add ceiling info if set
        if self._ceiling_gbp is not None:
            report["ceiling_gbp"] = self._ceiling_gbp
            report["remaining_gbp"] = round(self.remaining_gbp, 6)
            report["ceiling_reached"] = self.ceiling_reached

        return report

    def to_dict(self) -> dict:
        """Convert tracker state to dictionary."""
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self._started_at,
        }
        if self._ceiling_gbp is not None:
            data["ceiling_gbp"] = self._ceiling_gbp
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CostTracker":
        """Create tracker from dictionary."""
        ceiling_gbp = data.get("ceiling_gbp")
        tracker = cls(ceiling_gbp=ceiling_gbp)
        tracker._started_at = data.get("started_at", tracker._started_at)
        tracker.entries = [
            CostEntry.from_dict(e)
            for e in data.get("entries", [])
        ]
        return tracker

    def save(self, file_path: Path) -> None:
        """Save tracker state to file.

        Args:
            file_path: Path to save to
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(file_path, self.to_dict())

    @classmethod
    def load(cls, file_path: Path) -> "CostTracker":
        """Load tracker from file.

        Args:
            file_path: Path to load from

        Returns:
            Loaded CostTracker
        """
        data = read_json(file_path)
        return cls.from_dict(data)


# Pricing constants (USD per unit) - as of Jan 2026
class Pricing:
    """API pricing constants."""

    # Whisper API pricing (per minute)
    WHISPER_API_PER_MINUTE = 0.006

    # Claude Sonnet 4.5 pricing (per 1M tokens) - default
    CLAUDE_INPUT_PER_1M = 3.00
    CLAUDE_OUTPUT_PER_1M = 15.00

    # Claude Haiku 4.5 pricing (per 1M tokens)
    CLAUDE_HAIKU_INPUT_PER_1M = 1.00
    CLAUDE_HAIKU_OUTPUT_PER_1M = 5.00

    # OpenAI GPT-4.1 pricing (per 1M tokens) - default
    GPT4O_INPUT_PER_1M = 2.00
    GPT4O_OUTPUT_PER_1M = 8.00

    # OpenAI GPT-4.1 mini pricing (per 1M tokens)
    GPT4O_MINI_INPUT_PER_1M = 0.10
    GPT4O_MINI_OUTPUT_PER_1M = 0.40


@dataclass
class CostEstimate:
    """Estimated cost for an operation.

    Attributes:
        min_cost: Minimum estimated cost
        max_cost: Maximum estimated cost
        expected_cost: Expected (average) cost
        breakdown: Breakdown by category/service
        assumptions: Assumptions made for estimation
    """

    min_cost: float
    max_cost: float
    expected_cost: float
    breakdown: dict[str, float] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_cost": round(self.min_cost, 4),
            "max_cost": round(self.max_cost, 4),
            "expected_cost": round(self.expected_cost, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "assumptions": self.assumptions,
        }


class CostEstimator:
    """Estimates costs for processing operations."""

    def estimate_transcription_cost(
        self,
        duration_seconds: float,
        service: str = "whisper",
    ) -> float:
        """Estimate transcription cost.

        Args:
            duration_seconds: Audio duration in seconds
            service: Transcription service

        Returns:
            Estimated cost in USD
        """
        duration_minutes = duration_seconds / 60

        if service == "whisper":
            return duration_minutes * Pricing.WHISPER_API_PER_MINUTE

        return 0.0

    def estimate_llm_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        service: str = "claude",
        model: str = "default",
    ) -> float:
        """Estimate LLM cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            service: LLM service name
            model: Model name

        Returns:
            Estimated cost in USD
        """
        if service == "claude":
            if "haiku" in model.lower():
                input_cost = (input_tokens / 1_000_000) * Pricing.CLAUDE_HAIKU_INPUT_PER_1M
                output_cost = (output_tokens / 1_000_000) * Pricing.CLAUDE_HAIKU_OUTPUT_PER_1M
            else:
                input_cost = (input_tokens / 1_000_000) * Pricing.CLAUDE_INPUT_PER_1M
                output_cost = (output_tokens / 1_000_000) * Pricing.CLAUDE_OUTPUT_PER_1M
            return input_cost + output_cost

        if service == "openai":
            if "mini" in model.lower():
                input_cost = (input_tokens / 1_000_000) * Pricing.GPT4O_MINI_INPUT_PER_1M
                output_cost = (output_tokens / 1_000_000) * Pricing.GPT4O_MINI_OUTPUT_PER_1M
            else:
                input_cost = (input_tokens / 1_000_000) * Pricing.GPT4O_INPUT_PER_1M
                output_cost = (output_tokens / 1_000_000) * Pricing.GPT4O_OUTPUT_PER_1M
            return input_cost + output_cost

        return 0.0

    def estimate_highlight_processing(
        self,
        video_duration_seconds: float,
        service: str = "claude",
        model: str = "default",
    ) -> CostEstimate:
        """Estimate cost for highlight processing of one video.

        Args:
            video_duration_seconds: Video duration in seconds
            service: LLM service
            model: LLM model

        Returns:
            Cost estimate
        """
        # Transcription estimate
        transcription_cost = self.estimate_transcription_cost(video_duration_seconds)

        # LLM estimate
        # Assume ~150 words per minute, ~1.3 tokens per word
        words_per_minute = 150
        tokens_per_word = 1.3
        duration_minutes = video_duration_seconds / 60
        estimated_words = duration_minutes * words_per_minute
        estimated_input_tokens = int(estimated_words * tokens_per_word)

        # Output is typically smaller - highlights summary
        estimated_output_tokens = min(2000, estimated_input_tokens // 4)

        llm_cost = self.estimate_llm_cost(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            service=service,
            model=model,
        )

        total = transcription_cost + llm_cost

        return CostEstimate(
            min_cost=total * 0.7,  # 30% lower
            max_cost=total * 1.5,  # 50% higher
            expected_cost=total,
            breakdown={
                "transcription": transcription_cost,
                "llm_analysis": llm_cost,
            },
            assumptions=[
                f"Video duration: {video_duration_seconds:.0f}s",
                f"Estimated words: {estimated_words:.0f}",
                f"LLM service: {service}",
            ],
        )

    def estimate_batch_processing(
        self,
        video_durations: list[float],
        service: str = "claude",
        model: str = "default",
    ) -> CostEstimate:
        """Estimate cost for batch processing.

        Args:
            video_durations: List of video durations in seconds
            service: LLM service
            model: LLM model

        Returns:
            Cost estimate for entire batch
        """
        total_min = 0.0
        total_max = 0.0
        total_expected = 0.0
        transcription_total = 0.0
        llm_total = 0.0

        for duration in video_durations:
            estimate = self.estimate_highlight_processing(duration, service, model)
            total_min += estimate.min_cost
            total_max += estimate.max_cost
            total_expected += estimate.expected_cost
            transcription_total += estimate.breakdown.get("transcription", 0)
            llm_total += estimate.breakdown.get("llm_analysis", 0)

        return CostEstimate(
            min_cost=total_min,
            max_cost=total_max,
            expected_cost=total_expected,
            breakdown={
                "transcription": transcription_total,
                "llm_analysis": llm_total,
            },
            assumptions=[
                f"Video count: {len(video_durations)}",
                f"Total duration: {sum(video_durations):.0f}s",
                f"LLM service: {service}",
            ],
        )


def format_cost(amount: float) -> str:
    """Format a cost amount for display.

    Args:
        amount: Cost in USD

    Returns:
        Formatted string
    """
    if amount < 0.01:
        return f"${amount:.4f}"
    if amount < 1.00:
        return f"${amount:.3f}"
    return f"${amount:.2f}"


def format_cost_report(tracker: CostTracker) -> str:
    """Format a cost report for display.

    Args:
        tracker: Cost tracker with entries

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("COST REPORT")
    lines.append("=" * 50)

    # Total
    total = tracker.get_total()
    lines.append(f"\nTotal Cost: {format_cost(total)}")
    lines.append(f"Entries: {len(tracker.entries)}")

    # By category
    by_category = tracker.get_summary_by_category()
    if by_category:
        lines.append("\n--- By Category ---")
        for cat, summary in sorted(by_category.items()):
            lines.append(f"  {cat}: {format_cost(summary.total_amount)}")
            if summary.total_duration > 0:
                lines.append(f"    Duration: {summary.total_duration:.1f}s")
            if summary.total_tokens > 0:
                lines.append(f"    Tokens: {summary.total_tokens:,}")

    # By service
    by_service = tracker.get_summary_by_service()
    if by_service:
        lines.append("\n--- By Service ---")
        for service, summary in sorted(by_service.items()):
            lines.append(f"  {service}: {format_cost(summary.total_amount)} ({summary.entry_count} calls)")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)


def format_estimate(estimate: CostEstimate) -> str:
    """Format a cost estimate for display.

    Args:
        estimate: Cost estimate

    Returns:
        Formatted string
    """
    lines = []
    lines.append("Cost Estimate:")
    lines.append(f"  Expected: {format_cost(estimate.expected_cost)}")
    lines.append(f"  Range: {format_cost(estimate.min_cost)} - {format_cost(estimate.max_cost)}")

    if estimate.breakdown:
        lines.append("  Breakdown:")
        for service, amount in sorted(estimate.breakdown.items()):
            lines.append(f"    {service}: {format_cost(amount)}")

    if estimate.assumptions:
        lines.append("  Assumptions:")
        for assumption in estimate.assumptions:
            lines.append(f"    - {assumption}")

    return "\n".join(lines)
