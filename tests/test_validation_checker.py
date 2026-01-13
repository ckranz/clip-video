"""Tests for quality criteria checker."""

import pytest
from unittest.mock import Mock

from clip_video.validation.checker import QualityCriteriaChecker
from clip_video.validation.criteria import ValidationResult
from clip_video.modes.highlights import HighlightsConfig


class TestCheckDuration:
    """Tests for duration checking."""

    def test_check_duration_too_short(self):
        """Test that clips shorter than min_duration fail."""
        config = HighlightsConfig()  # min_duration=30.0
        checker = QualityCriteriaChecker(config)

        result = checker.check_duration(start=0.0, end=20.0)

        assert result.result == ValidationResult.FAIL
        assert "Too short" in result.reason
        assert "20.0s" in result.reason
        assert result.details["actual_duration"] == 20.0

    def test_check_duration_too_long(self):
        """Test that clips longer than max_duration fail."""
        config = HighlightsConfig()  # max_duration=120.0
        checker = QualityCriteriaChecker(config)

        result = checker.check_duration(start=0.0, end=150.0)

        assert result.result == ValidationResult.FAIL
        assert "Too long" in result.reason
        assert "150.0s" in result.reason

    def test_check_duration_valid(self):
        """Test that clips within duration range pass."""
        config = HighlightsConfig()  # 30-120s
        checker = QualityCriteriaChecker(config)

        result = checker.check_duration(start=0.0, end=60.0)

        assert result.result == ValidationResult.PASS
        assert "Duration OK" in result.reason
        assert "60.0s" in result.reason

    def test_check_duration_at_minimum(self):
        """Test clip exactly at minimum duration passes."""
        config = HighlightsConfig()  # min_duration=30.0
        checker = QualityCriteriaChecker(config)

        result = checker.check_duration(start=0.0, end=30.0)

        assert result.result == ValidationResult.PASS

    def test_check_duration_at_maximum(self):
        """Test clip exactly at maximum duration passes."""
        config = HighlightsConfig()  # max_duration=120.0
        checker = QualityCriteriaChecker(config)

        result = checker.check_duration(start=0.0, end=120.0)

        assert result.result == ValidationResult.PASS

    def test_check_duration_custom_config(self):
        """Test with custom duration constraints."""
        config = HighlightsConfig(min_duration=45.0, max_duration=90.0)
        checker = QualityCriteriaChecker(config)

        # 40s is too short for 45s minimum
        result = checker.check_duration(start=0.0, end=40.0)
        assert result.result == ValidationResult.FAIL

        # 60s is within range
        result = checker.check_duration(start=0.0, end=60.0)
        assert result.result == ValidationResult.PASS


class TestCheckOverlap:
    """Tests for overlap detection."""

    def test_check_overlap_no_used_segments(self):
        """Test with no previously used segments."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        result = checker.check_overlap(start=10.0, end=40.0)

        assert result.result == ValidationResult.PASS
        assert "No overlap" in result.reason

    def test_check_overlap_detected(self):
        """Test overlap detection with used segment."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)

        # New segment overlaps: 30-60s overlaps with 10-40s
        result = checker.check_overlap(start=30.0, end=60.0)

        assert result.result == ValidationResult.FAIL
        assert "Overlaps" in result.reason
        assert "10.0-40.0s" in result.reason

    def test_check_overlap_adjacent_ok(self):
        """Test that adjacent (non-overlapping) segments pass."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)

        # Segment starts exactly where previous ends
        result = checker.check_overlap(start=40.0, end=70.0)

        assert result.result == ValidationResult.PASS

    def test_check_overlap_before_used_segment(self):
        """Test segment entirely before used segment passes."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(50.0, 80.0)

        result = checker.check_overlap(start=10.0, end=40.0)

        assert result.result == ValidationResult.PASS

    def test_check_overlap_after_used_segment(self):
        """Test segment entirely after used segment passes."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)

        result = checker.check_overlap(start=50.0, end=80.0)

        assert result.result == ValidationResult.PASS

    def test_check_overlap_contained_within(self):
        """Test segment contained within used segment fails."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 80.0)

        # New segment is inside the used segment
        result = checker.check_overlap(start=30.0, end=50.0)

        assert result.result == ValidationResult.FAIL

    def test_check_overlap_contains_used_segment(self):
        """Test segment that contains used segment fails."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(30.0, 50.0)

        # New segment wraps around the used segment
        result = checker.check_overlap(start=10.0, end=80.0)

        assert result.result == ValidationResult.FAIL

    def test_check_overlap_multiple_segments(self):
        """Test overlap detection with multiple used segments."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)
        checker.mark_segment_used(60.0, 90.0)
        checker.mark_segment_used(100.0, 130.0)

        # Overlaps with second segment
        result = checker.check_overlap(start=50.0, end=70.0)
        assert result.result == ValidationResult.FAIL
        assert "60.0-90.0s" in result.reason

        # No overlap - between segments
        result = checker.check_overlap(start=42.0, end=58.0)
        assert result.result == ValidationResult.PASS


class TestMarkSegmentUsed:
    """Tests for segment tracking."""

    def test_mark_segment_used(self):
        """Test marking a segment as used."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        checker.mark_segment_used(10.0, 40.0)

        assert len(checker.used_segments) == 1
        assert checker.used_segments[0] == (10.0, 40.0)

    def test_mark_multiple_segments(self):
        """Test marking multiple segments."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        checker.mark_segment_used(10.0, 40.0)
        checker.mark_segment_used(50.0, 80.0)

        assert len(checker.used_segments) == 2

    def test_clear_used_segments(self):
        """Test clearing tracked segments."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)
        checker.mark_segment_used(50.0, 80.0)

        checker.clear_used_segments()

        assert len(checker.used_segments) == 0


class TestRunDeterministicChecks:
    """Tests for running all deterministic checks."""

    def test_run_deterministic_checks_all_pass(self):
        """Test all checks pass for valid clip."""
        config = HighlightsConfig()  # 30-120s
        checker = QualityCriteriaChecker(config)

        results = checker.run_deterministic_checks(start=0.0, end=60.0)

        assert len(results) == 2  # duration and overlap
        assert all(r.result == ValidationResult.PASS for r in results)

    def test_run_deterministic_checks_duration_fails(self):
        """Test duration check fails, overlap passes."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        results = checker.run_deterministic_checks(start=0.0, end=20.0)

        duration_result = next(r for r in results if r.criterion == "duration")
        overlap_result = next(r for r in results if r.criterion == "overlap")

        assert duration_result.result == ValidationResult.FAIL
        assert overlap_result.result == ValidationResult.PASS

    def test_run_deterministic_checks_overlap_fails(self):
        """Test overlap check fails, duration passes."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(30.0, 90.0)

        results = checker.run_deterministic_checks(start=60.0, end=120.0)

        duration_result = next(r for r in results if r.criterion == "duration")
        overlap_result = next(r for r in results if r.criterion == "overlap")

        assert duration_result.result == ValidationResult.PASS
        assert overlap_result.result == ValidationResult.FAIL

    def test_run_deterministic_checks_multiple_failures(self):
        """Test both checks can fail."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(10.0, 40.0)

        # Too short (20s) AND overlaps (20-40 overlaps with 10-40)
        results = checker.run_deterministic_checks(start=20.0, end=40.0)

        assert len(results) == 2
        assert all(r.result == ValidationResult.FAIL for r in results)


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_all_deterministic_passed_true(self):
        """Test all_deterministic_passed returns True when all pass."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        assert checker.all_deterministic_passed(start=0.0, end=60.0) is True

    def test_all_deterministic_passed_false(self):
        """Test all_deterministic_passed returns False when any fails."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        assert checker.all_deterministic_passed(start=0.0, end=20.0) is False

    def test_get_deterministic_failures_none(self):
        """Test get_deterministic_failures returns empty list when all pass."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        failures = checker.get_deterministic_failures(start=0.0, end=60.0)

        assert len(failures) == 0

    def test_get_deterministic_failures_some(self):
        """Test get_deterministic_failures returns only failed checks."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)
        checker.mark_segment_used(30.0, 90.0)

        # Duration OK (60s) but overlaps
        failures = checker.get_deterministic_failures(start=60.0, end=120.0)

        assert len(failures) == 1
        assert failures[0].criterion == "overlap"


class TestBrandContext:
    """Tests for brand context handling."""

    def test_brand_context_defaults_empty(self):
        """Test brand_context defaults to empty dict."""
        config = HighlightsConfig()
        checker = QualityCriteriaChecker(config)

        assert checker.brand_context == {}

    def test_brand_context_stored(self):
        """Test brand_context is stored correctly."""
        config = HighlightsConfig()
        context = {"name": "KCD", "keywords": ["kubernetes", "cloud native"]}
        checker = QualityCriteriaChecker(config, brand_context=context)

        assert checker.brand_context["name"] == "KCD"
        assert "kubernetes" in checker.brand_context["keywords"]
