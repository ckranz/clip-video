"""Tests for validation criteria data structures."""

import pytest

from clip_video.validation.criteria import (
    ValidationResult,
    CriterionResult,
    ClipValidation,
)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_values(self):
        """Test enum values."""
        assert ValidationResult.PASS.value == "pass"
        assert ValidationResult.FAIL.value == "fail"
        assert ValidationResult.WARN.value == "warn"

    def test_string_conversion(self):
        """Test string representation via .value."""
        # The .value property returns the string
        assert ValidationResult.PASS.value == "pass"
        assert ValidationResult.FAIL.value == "fail"
        # Can compare directly due to str inheritance
        assert ValidationResult.PASS == "pass"
        assert ValidationResult.FAIL == "fail"


class TestCriterionResult:
    """Tests for CriterionResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        result = CriterionResult(
            criterion="duration",
            result=ValidationResult.PASS,
            reason="Duration OK: 45.0s",
        )

        assert result.criterion == "duration"
        assert result.result == ValidationResult.PASS
        assert result.reason == "Duration OK: 45.0s"
        assert result.details == {}

    def test_with_details(self):
        """Test creation with details."""
        result = CriterionResult(
            criterion="duration",
            result=ValidationResult.FAIL,
            reason="Too short: 15s < 30s",
            details={"actual": 15.0, "minimum": 30.0},
        )

        assert result.details["actual"] == 15.0
        assert result.details["minimum"] == 30.0

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict maintain data."""
        original = CriterionResult(
            criterion="overlap",
            result=ValidationResult.FAIL,
            reason="Overlaps with segment 10.0-40.0s",
            details={"overlap_start": 10.0, "overlap_end": 40.0},
        )

        data = original.to_dict()
        loaded = CriterionResult.from_dict(data)

        assert loaded.criterion == original.criterion
        assert loaded.result == original.result
        assert loaded.reason == original.reason
        assert loaded.details == original.details


class TestClipValidation:
    """Tests for ClipValidation dataclass."""

    def test_passed_property(self):
        """Test passed property returns True for PASS result."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=40.0,
            overall_result=ValidationResult.PASS,
            criteria_results=[],
            transcript_segment="Test text",
        )

        assert validation.passed is True
        assert validation.failed is False

    def test_failed_property(self):
        """Test failed property returns True for FAIL result."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=40.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[],
            transcript_segment="Test text",
        )

        assert validation.passed is False
        assert validation.failed is True

    def test_duration_property(self):
        """Test duration calculation."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=70.0,
            overall_result=ValidationResult.PASS,
            criteria_results=[],
            transcript_segment="Test text",
        )

        assert validation.duration == 60.0

    def test_failure_reasons(self):
        """Test failure_reasons extracts only failed criteria reasons."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=40.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[
                CriterionResult(
                    criterion="duration",
                    result=ValidationResult.PASS,
                    reason="Duration OK",
                ),
                CriterionResult(
                    criterion="overlap",
                    result=ValidationResult.FAIL,
                    reason="Overlaps with approved clip",
                ),
                CriterionResult(
                    criterion="sentence_boundaries",
                    result=ValidationResult.FAIL,
                    reason="Cuts mid-sentence",
                ),
                CriterionResult(
                    criterion="hooks",
                    result=ValidationResult.WARN,
                    reason="No hook detected",
                ),
            ],
            transcript_segment="Test text",
        )

        reasons = validation.failure_reasons
        assert len(reasons) == 2
        assert "Overlaps with approved clip" in reasons
        assert "Cuts mid-sentence" in reasons
        assert "Duration OK" not in reasons
        assert "No hook detected" not in reasons

    def test_failed_criteria(self):
        """Test failed_criteria extracts criterion names."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=40.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[
                CriterionResult(
                    criterion="duration",
                    result=ValidationResult.PASS,
                    reason="OK",
                ),
                CriterionResult(
                    criterion="overlap",
                    result=ValidationResult.FAIL,
                    reason="Failed",
                ),
            ],
            transcript_segment="Test text",
        )

        failed = validation.failed_criteria
        assert "overlap" in failed
        assert "duration" not in failed

    def test_warning_reasons(self):
        """Test warning_reasons extracts only warnings."""
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=40.0,
            overall_result=ValidationResult.PASS,
            criteria_results=[
                CriterionResult(
                    criterion="hooks",
                    result=ValidationResult.WARN,
                    reason="No hook detected (optional)",
                ),
            ],
            transcript_segment="Test text",
        )

        warnings = validation.warning_reasons
        assert len(warnings) == 1
        assert "No hook detected" in warnings[0]

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict maintain data."""
        original = ClipValidation(
            clip_id="clip_01",
            start_time=10.0,
            end_time=70.0,
            overall_result=ValidationResult.PASS,
            criteria_results=[
                CriterionResult(
                    criterion="duration",
                    result=ValidationResult.PASS,
                    reason="Duration OK: 60s",
                ),
            ],
            transcript_segment="This is the transcript segment.",
        )

        data = original.to_dict()

        # Check serialized data
        assert data["clip_id"] == "clip_01"
        assert data["duration"] == 60.0
        assert data["passed"] is True
        assert len(data["criteria_results"]) == 1

        # Roundtrip
        loaded = ClipValidation.from_dict(data)

        assert loaded.clip_id == original.clip_id
        assert loaded.start_time == original.start_time
        assert loaded.end_time == original.end_time
        assert loaded.overall_result == original.overall_result
        assert loaded.transcript_segment == original.transcript_segment
        assert len(loaded.criteria_results) == 1
        assert loaded.criteria_results[0].criterion == "duration"

    def test_create_failed_convenience(self):
        """Test create_failed convenience method."""
        results = [
            CriterionResult(
                criterion="duration",
                result=ValidationResult.FAIL,
                reason="Too short",
            ),
        ]

        validation = ClipValidation.create_failed(
            clip_id="clip_01",
            start_time=0.0,
            end_time=20.0,
            transcript_segment="Short segment",
            criteria_results=results,
        )

        assert validation.overall_result == ValidationResult.FAIL
        assert validation.failed is True
        assert validation.passed is False

    def test_create_passed_convenience(self):
        """Test create_passed convenience method."""
        results = [
            CriterionResult(
                criterion="duration",
                result=ValidationResult.PASS,
                reason="OK",
            ),
        ]

        validation = ClipValidation.create_passed(
            clip_id="clip_01",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Good segment",
            criteria_results=results,
        )

        assert validation.overall_result == ValidationResult.PASS
        assert validation.passed is True
        assert validation.failed is False
