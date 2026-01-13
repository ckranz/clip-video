"""Validation data structures for clip quality checking.

Defines the core types used to represent validation results:
- ValidationResult: Pass/Fail/Warn status
- CriterionResult: Result for a single quality criterion
- ClipValidation: Complete validation result for a clip
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationResult(str, Enum):
    """Status of a validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class CriterionResult:
    """Result of checking a single quality criterion.

    Attributes:
        criterion: Name of the criterion checked (e.g., "duration", "overlap")
        result: Whether the check passed, failed, or warned
        reason: Human-readable explanation of the result
        details: Optional additional details about the check
    """

    criterion: str
    result: ValidationResult
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "criterion": self.criterion,
            "result": self.result.value,
            "reason": self.reason,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionResult":
        """Create from dictionary."""
        return cls(
            criterion=data["criterion"],
            result=ValidationResult(data["result"]),
            reason=data["reason"],
            details=data.get("details", {}),
        )


@dataclass
class ClipValidation:
    """Complete validation result for a clip.

    Aggregates results from all quality criteria checks for a single clip.

    Attributes:
        clip_id: Unique identifier for the clip
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        overall_result: Aggregated pass/fail/warn status
        criteria_results: Results for each individual criterion
        transcript_segment: The transcript text for this clip segment
    """

    clip_id: str
    start_time: float
    end_time: float
    overall_result: ValidationResult
    criteria_results: list[CriterionResult]
    transcript_segment: str

    @property
    def passed(self) -> bool:
        """Check if the clip passed validation."""
        return self.overall_result == ValidationResult.PASS

    @property
    def failed(self) -> bool:
        """Check if the clip failed validation."""
        return self.overall_result == ValidationResult.FAIL

    @property
    def duration(self) -> float:
        """Calculate clip duration in seconds."""
        return self.end_time - self.start_time

    @property
    def failure_reasons(self) -> list[str]:
        """Get list of reasons why the clip failed."""
        return [
            r.reason for r in self.criteria_results
            if r.result == ValidationResult.FAIL
        ]

    @property
    def warning_reasons(self) -> list[str]:
        """Get list of warning messages."""
        return [
            r.reason for r in self.criteria_results
            if r.result == ValidationResult.WARN
        ]

    @property
    def failed_criteria(self) -> list[str]:
        """Get list of criteria that failed."""
        return [
            r.criterion for r in self.criteria_results
            if r.result == ValidationResult.FAIL
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "clip_id": self.clip_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "overall_result": self.overall_result.value,
            "passed": self.passed,
            "criteria_results": [r.to_dict() for r in self.criteria_results],
            "failure_reasons": self.failure_reasons,
            "transcript_segment": self.transcript_segment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipValidation":
        """Create from dictionary."""
        return cls(
            clip_id=data["clip_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            overall_result=ValidationResult(data["overall_result"]),
            criteria_results=[
                CriterionResult.from_dict(r) for r in data["criteria_results"]
            ],
            transcript_segment=data["transcript_segment"],
        )

    @classmethod
    def create_failed(
        cls,
        clip_id: str,
        start_time: float,
        end_time: float,
        transcript_segment: str,
        criteria_results: list[CriterionResult],
    ) -> "ClipValidation":
        """Create a failed validation result.

        Convenience method for creating a ClipValidation with FAIL status.
        """
        return cls(
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            overall_result=ValidationResult.FAIL,
            criteria_results=criteria_results,
            transcript_segment=transcript_segment,
        )

    @classmethod
    def create_passed(
        cls,
        clip_id: str,
        start_time: float,
        end_time: float,
        transcript_segment: str,
        criteria_results: list[CriterionResult],
    ) -> "ClipValidation":
        """Create a passed validation result.

        Convenience method for creating a ClipValidation with PASS status.
        """
        return cls(
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            overall_result=ValidationResult.PASS,
            criteria_results=criteria_results,
            transcript_segment=transcript_segment,
        )
