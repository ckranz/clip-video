"""Tests for validation orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from clip_video.validation.orchestrator import (
    AgenticValidator,
    ValidationPass,
    RunSummary,
)
from clip_video.validation.criteria import ValidationResult, ClipValidation
from clip_video.llm.base import (
    HighlightSegment,
    ClipValidationRequest,
    ClipValidationResponse,
)
from clip_video.modes.highlights import HighlightsConfig


class TestValidationPass:
    """Tests for ValidationPass dataclass."""

    def test_basic_creation(self):
        """Test basic creation of ValidationPass."""
        vp = ValidationPass(
            approved=[],
            rejected=[],
        )

        assert vp.approved == []
        assert vp.rejected == []
        assert vp.exhausted is False
        assert vp.cost_ceiling_hit is False
        assert vp.iterations == 0

    def test_with_clips(self):
        """Test ValidationPass with actual clips."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=70.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )
        validation = ClipValidation(
            clip_id="clip_01",
            start_time=0.0,
            end_time=30.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[],
            transcript_segment="Test",
        )

        vp = ValidationPass(
            approved=[segment],
            rejected=[(segment, validation)],
            iterations=3,
        )

        assert len(vp.approved) == 1
        assert len(vp.rejected) == 1
        assert vp.iterations == 3


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_target_met(self):
        """Test summary when target is met."""
        summary = RunSummary(
            approved_count=5,
            rejected_count=2,
            target_clips=5,
            min_acceptable=3,
            target_met=True,
            minimum_met=True,
            iterations=7,
            total_cost_usd=0.05,
            total_cost_gbp=0.04,
            cost_ceiling_gbp=5.0,
            cost_ceiling_hit=False,
            transcript_exhausted=False,
            termination_reason="Target clip count reached",
        )

        assert summary.target_met is True
        assert summary.minimum_met is True
        assert summary.approved_count == 5

    def test_minimum_met(self):
        """Test summary when only minimum is met."""
        summary = RunSummary(
            approved_count=3,
            rejected_count=5,
            target_clips=5,
            min_acceptable=3,
            target_met=False,
            minimum_met=True,
            iterations=10,
            total_cost_usd=0.10,
            total_cost_gbp=0.08,
            cost_ceiling_gbp=5.0,
            cost_ceiling_hit=False,
            transcript_exhausted=True,
            termination_reason="Transcript exhausted",
        )

        assert summary.target_met is False
        assert summary.minimum_met is True
        assert summary.transcript_exhausted is True


class TestAgenticValidatorInit:
    """Tests for AgenticValidator initialization."""

    def test_basic_init(self, tmp_path):
        """Test basic initialization."""
        config = HighlightsConfig()
        mock_llm = Mock()

        validator = AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test transcript",
            project_root=tmp_path,
        )

        assert validator.config == config
        assert validator.llm == mock_llm
        assert validator.transcript_text == "Test transcript"
        assert validator.approved_clips == []
        assert validator.used_segments == []

    def test_init_with_brand_context(self, tmp_path):
        """Test initialization with brand context."""
        config = HighlightsConfig()
        mock_llm = Mock()
        brand_ctx = {"channel": "Tech Channel", "topics": ["AI"]}

        validator = AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test",
            brand_context=brand_ctx,
            project_root=tmp_path,
        )

        assert validator.brand_context == brand_ctx


class TestAgenticValidatorValidation:
    """Tests for clip validation."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator with mock LLM."""
        config = HighlightsConfig()
        mock_llm = Mock()

        # Setup mock validate_clip to return valid
        mock_llm.validate_clip = Mock(return_value=ClipValidationResponse(
            clip_id="test",
            is_valid=True,
            sentence_boundaries_ok=True,
            topic_complete=True,
            has_hook=True,
            standalone_valid=True,
            brand_relevant=True,
            transcript_aligned=True,
            tokens_used=100,
        ))

        return AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="This is a test transcript.",
            project_root=tmp_path,
        )

    def test_validate_clip_passes_deterministic(self, validator):
        """Test that valid clip passes deterministic checks."""
        segment = HighlightSegment(
            start_time=0.0,
            end_time=60.0,  # 60s is within 30-120s range
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = validator._validate_clip(segment)

        # Should pass deterministic and call LLM
        assert validation.overall_result == ValidationResult.PASS
        validator.llm.validate_clip.assert_called_once()

    def test_validate_clip_fails_duration_too_short(self, validator):
        """Test that too-short clip fails deterministic check."""
        segment = HighlightSegment(
            start_time=0.0,
            end_time=20.0,  # 20s is below 30s minimum
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = validator._validate_clip(segment)

        # Should fail deterministic check, LLM not called
        assert validation.overall_result == ValidationResult.FAIL
        assert "duration" in validation.failed_criteria
        validator.llm.validate_clip.assert_not_called()

    def test_validate_clip_fails_duration_too_long(self, validator):
        """Test that too-long clip fails deterministic check."""
        segment = HighlightSegment(
            start_time=0.0,
            end_time=150.0,  # 150s is above 120s maximum
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = validator._validate_clip(segment)

        assert validation.overall_result == ValidationResult.FAIL
        assert "duration" in validation.failed_criteria

    def test_validate_clip_fails_overlap(self, validator):
        """Test that overlapping clip fails."""
        # Mark a segment as used
        validator.checker.mark_segment_used(0.0, 60.0)

        segment = HighlightSegment(
            start_time=30.0,  # Overlaps with 0-60
            end_time=90.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = validator._validate_clip(segment)

        assert validation.overall_result == ValidationResult.FAIL
        assert "overlap" in validation.failed_criteria

    def test_validate_clip_llm_fails(self, validator):
        """Test that LLM validation failure is handled."""
        # Setup mock to return invalid
        validator.llm.validate_clip = Mock(return_value=ClipValidationResponse(
            clip_id="test",
            is_valid=False,
            sentence_boundaries_ok=False,
            topic_complete=True,
            has_hook=False,
            standalone_valid=True,
            brand_relevant=True,
            transcript_aligned=True,
            issues=["Starts mid-sentence"],
            tokens_used=100,
        ))

        segment = HighlightSegment(
            start_time=0.0,
            end_time=60.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = validator._validate_clip(segment)

        assert validation.overall_result == ValidationResult.FAIL


class TestAgenticValidatorApproval:
    """Tests for clip approval."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator."""
        config = HighlightsConfig()
        mock_llm = Mock()
        return AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test",
            project_root=tmp_path,
        )

    def test_approve_clip(self, validator):
        """Test approving a clip."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=70.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validator._approve_clip(segment)

        assert len(validator.approved_clips) == 1
        assert validator.approved_clips[0] == segment
        assert (10.0, 70.0) in validator.used_segments


class TestAgenticValidatorRejection:
    """Tests for clip rejection."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator."""
        config = HighlightsConfig()
        mock_llm = Mock()
        return AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test",
            project_root=tmp_path,
            video_path="/path/to/video.mp4",
        )

    def test_reject_clip(self, validator):
        """Test rejecting a clip adds to review queue."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=70.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = ClipValidation(
            clip_id="clip_10.0_70.0",
            start_time=10.0,
            end_time=70.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[],
            transcript_segment="Test",
        )

        validator._reject_clip(segment, validation)

        assert validator.review_queue.count() == 1
        assert validator.rejection_counts.get("clip_10.0_70.0") == 1

    def test_reject_clip_increments_count(self, validator):
        """Test that rejection count increments."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=70.0,
            summary="Test clip",
            hook_text="Hook",
            reason="Good content",
        )

        validation = ClipValidation(
            clip_id="clip_10.0_70.0",
            start_time=10.0,
            end_time=70.0,
            overall_result=ValidationResult.FAIL,
            criteria_results=[],
            transcript_segment="Test",
        )

        # Reject twice
        validator._reject_clip(segment, validation)
        validator._reject_clip(segment, validation)

        assert validator.rejection_counts.get("clip_10.0_70.0") == 2


class TestAgenticValidatorWorkflow:
    """Integration tests for the full workflow."""

    @pytest.fixture
    def setup_validator(self, tmp_path):
        """Create a validator with mock LLM that tracks calls."""
        config = HighlightsConfig(target_clips=3, min_acceptable_clips=2)
        mock_llm = Mock()

        # Track calls
        call_count = {"validate": 0, "replace": 0}

        def mock_validate(request):
            call_count["validate"] += 1
            return ClipValidationResponse(
                clip_id=request.clip_id,
                is_valid=True,
                tokens_used=100,
            )

        def mock_replace(rejected_clips, transcript_text, used_segments, target_count):
            call_count["replace"] += 1
            return []  # No replacements found

        mock_llm.validate_clip = Mock(side_effect=mock_validate)
        mock_llm.find_replacement_clips = Mock(side_effect=mock_replace)

        validator = AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test transcript",
            project_root=tmp_path,
        )

        return validator, call_count

    def test_validate_and_refine_all_pass(self, setup_validator):
        """Test workflow where all clips pass."""
        validator, call_count = setup_validator

        segments = [
            HighlightSegment(start_time=0.0, end_time=60.0, summary="Clip 1", hook_text="H1", reason="R1"),
            HighlightSegment(start_time=70.0, end_time=130.0, summary="Clip 2", hook_text="H2", reason="R2"),
            HighlightSegment(start_time=140.0, end_time=200.0, summary="Clip 3", hook_text="H3", reason="R3"),
        ]

        approved, summary = validator.validate_and_refine(segments)

        assert len(approved) == 3
        assert summary.target_met is True
        assert summary.approved_count == 3
        assert call_count["validate"] == 3

    def test_validate_and_refine_partial_pass(self, setup_validator):
        """Test workflow where some clips fail deterministic checks."""
        validator, call_count = setup_validator

        segments = [
            HighlightSegment(start_time=0.0, end_time=60.0, summary="Clip 1", hook_text="H1", reason="R1"),
            HighlightSegment(start_time=70.0, end_time=80.0, summary="Clip 2 (too short)", hook_text="H2", reason="R2"),  # 10s - fails
            HighlightSegment(start_time=140.0, end_time=200.0, summary="Clip 3", hook_text="H3", reason="R3"),
        ]

        approved, summary = validator.validate_and_refine(segments)

        # Only 2 approved (one too short)
        assert len(approved) == 2
        assert summary.approved_count == 2
        assert summary.rejected_count >= 1
        # Should try to find replacements
        assert call_count["replace"] >= 1

    def test_validate_and_refine_exhausted(self, setup_validator):
        """Test workflow where transcript is exhausted."""
        validator, call_count = setup_validator

        # Only 2 valid clips, need 3
        segments = [
            HighlightSegment(start_time=0.0, end_time=60.0, summary="Clip 1", hook_text="H1", reason="R1"),
            HighlightSegment(start_time=70.0, end_time=130.0, summary="Clip 2", hook_text="H2", reason="R2"),
        ]

        approved, summary = validator.validate_and_refine(segments)

        assert len(approved) == 2
        assert summary.target_met is False
        assert summary.minimum_met is True
        assert summary.transcript_exhausted is True


class TestLLMToCriteria:
    """Tests for converting LLM response to criteria results."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator."""
        config = HighlightsConfig()
        mock_llm = Mock()
        return AgenticValidator(
            config=config,
            llm=mock_llm,
            transcript_text="Test",
            project_root=tmp_path,
        )

    def test_all_pass(self, validator):
        """Test conversion when all criteria pass."""
        response = ClipValidationResponse(
            clip_id="test",
            is_valid=True,
            sentence_boundaries_ok=True,
            topic_complete=True,
            has_hook=True,
            standalone_valid=True,
            brand_relevant=True,
            transcript_aligned=True,
        )

        criteria = validator._llm_to_criteria(response)

        assert len(criteria) == 6
        assert all(c.result in [ValidationResult.PASS, ValidationResult.WARN] for c in criteria)

    def test_some_fail(self, validator):
        """Test conversion when some criteria fail."""
        response = ClipValidationResponse(
            clip_id="test",
            is_valid=False,
            sentence_boundaries_ok=False,
            topic_complete=True,
            has_hook=False,
            standalone_valid=False,
            brand_relevant=True,
            transcript_aligned=True,
        )

        criteria = validator._llm_to_criteria(response)

        # Check specific criteria
        sentence_result = next(c for c in criteria if c.criterion == "sentence_boundaries")
        assert sentence_result.result == ValidationResult.FAIL

        hook_result = next(c for c in criteria if c.criterion == "has_hook")
        assert hook_result.result == ValidationResult.WARN  # Optional

        standalone_result = next(c for c in criteria if c.criterion == "standalone_valid")
        assert standalone_result.result == ValidationResult.FAIL
