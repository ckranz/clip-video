"""Validation module for multi-pass clip quality checking.

This module provides data structures and logic for validating highlight
clips against quality criteria before approval.
"""

from clip_video.validation.criteria import (
    ValidationResult,
    CriterionResult,
    ClipValidation,
)
from clip_video.validation.checker import QualityCriteriaChecker
from clip_video.validation.orchestrator import (
    AgenticValidator,
    ValidationPass,
    RunSummary,
)

__all__ = [
    "ValidationResult",
    "CriterionResult",
    "ClipValidation",
    "QualityCriteriaChecker",
    "AgenticValidator",
    "ValidationPass",
    "RunSummary",
]
