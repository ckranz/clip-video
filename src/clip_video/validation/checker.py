"""Quality criteria checker for clip validation.

Implements deterministic quality checks that can run before LLM validation:
- Duration constraints (D001)
- Overlap detection (D008/D013)

These checks are fast and don't require LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from clip_video.validation.criteria import CriterionResult, ValidationResult

if TYPE_CHECKING:
    from clip_video.modes.highlights import HighlightsConfig


class QualityCriteriaChecker:
    """Checks clips against deterministic quality criteria.

    Performs checks that don't require LLM analysis:
    - Duration validation (30-120s for YouTube Shorts)
    - Overlap detection with already-approved clips

    Attributes:
        config: Highlights configuration with duration constraints
        brand_context: Optional brand context for relevance checking
        used_segments: List of (start, end) tuples for approved segments
    """

    def __init__(
        self,
        config: "HighlightsConfig",
        brand_context: dict | None = None,
    ):
        """Initialize the checker.

        Args:
            config: Highlights configuration
            brand_context: Optional brand context dictionary
        """
        self.config = config
        self.brand_context = brand_context or {}
        self.used_segments: list[tuple[float, float]] = []

    def check_duration(self, start: float, end: float) -> CriterionResult:
        """Check if clip duration is within constraints (D001).

        Args:
            start: Clip start time in seconds
            end: Clip end time in seconds

        Returns:
            CriterionResult with pass/fail status
        """
        duration = end - start

        if duration < self.config.min_duration:
            return CriterionResult(
                criterion="duration",
                result=ValidationResult.FAIL,
                reason=f"Too short: {duration:.1f}s < {self.config.min_duration:.0f}s minimum",
                details={
                    "actual_duration": duration,
                    "min_duration": self.config.min_duration,
                    "max_duration": self.config.max_duration,
                },
            )

        if duration > self.config.max_duration:
            return CriterionResult(
                criterion="duration",
                result=ValidationResult.FAIL,
                reason=f"Too long: {duration:.1f}s > {self.config.max_duration:.0f}s maximum",
                details={
                    "actual_duration": duration,
                    "min_duration": self.config.min_duration,
                    "max_duration": self.config.max_duration,
                },
            )

        return CriterionResult(
            criterion="duration",
            result=ValidationResult.PASS,
            reason=f"Duration OK: {duration:.1f}s (within {self.config.min_duration:.0f}-{self.config.max_duration:.0f}s)",
            details={
                "actual_duration": duration,
                "min_duration": self.config.min_duration,
                "max_duration": self.config.max_duration,
            },
        )

    def check_overlap(self, start: float, end: float) -> CriterionResult:
        """Check if clip overlaps with already-approved clips (D008/D013).

        Args:
            start: Clip start time in seconds
            end: Clip end time in seconds

        Returns:
            CriterionResult with pass/fail status
        """
        for used_start, used_end in self.used_segments:
            # Check for any overlap
            if start < used_end and end > used_start:
                # Calculate overlap details
                overlap_start = max(start, used_start)
                overlap_end = min(end, used_end)
                overlap_duration = overlap_end - overlap_start

                return CriterionResult(
                    criterion="overlap",
                    result=ValidationResult.FAIL,
                    reason=f"Overlaps with approved segment {used_start:.1f}-{used_end:.1f}s",
                    details={
                        "overlap_with": {"start": used_start, "end": used_end},
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_duration": overlap_duration,
                    },
                )

        return CriterionResult(
            criterion="overlap",
            result=ValidationResult.PASS,
            reason="No overlap with approved clips",
            details={"checked_segments": len(self.used_segments)},
        )

    def mark_segment_used(self, start: float, end: float) -> None:
        """Track an approved segment to prevent future overlaps.

        Args:
            start: Segment start time in seconds
            end: Segment end time in seconds
        """
        self.used_segments.append((start, end))

    def clear_used_segments(self) -> None:
        """Clear all tracked segments."""
        self.used_segments.clear()

    def run_deterministic_checks(
        self,
        start: float,
        end: float,
    ) -> list[CriterionResult]:
        """Run all deterministic (non-LLM) checks on a clip.

        Args:
            start: Clip start time in seconds
            end: Clip end time in seconds

        Returns:
            List of CriterionResult for each check performed
        """
        return [
            self.check_duration(start, end),
            self.check_overlap(start, end),
        ]

    def all_deterministic_passed(
        self,
        start: float,
        end: float,
    ) -> bool:
        """Check if all deterministic checks pass.

        Args:
            start: Clip start time in seconds
            end: Clip end time in seconds

        Returns:
            True if all deterministic checks pass
        """
        results = self.run_deterministic_checks(start, end)
        return all(r.result == ValidationResult.PASS for r in results)

    def get_deterministic_failures(
        self,
        start: float,
        end: float,
    ) -> list[CriterionResult]:
        """Get only the failed deterministic checks.

        Args:
            start: Clip start time in seconds
            end: Clip end time in seconds

        Returns:
            List of failed CriterionResult
        """
        results = self.run_deterministic_checks(start, end)
        return [r for r in results if r.result == ValidationResult.FAIL]
