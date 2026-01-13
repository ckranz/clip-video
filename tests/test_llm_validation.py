"""Tests for LLM validation request/response data structures."""

import pytest
from clip_video.llm.base import ClipValidationRequest, ClipValidationResponse


class TestClipValidationRequest:
    """Tests for ClipValidationRequest dataclass."""

    def test_basic_creation(self):
        """Test basic request creation."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="This is the transcript text for testing.",
            start_time=10.0,
            end_time=70.0,
            clip_summary="A test clip about something interesting.",
        )

        assert request.clip_id == "clip_01"
        assert request.transcript_segment == "This is the transcript text for testing."
        assert request.start_time == 10.0
        assert request.end_time == 70.0
        assert request.clip_summary == "A test clip about something interesting."

    def test_default_quality_criteria(self):
        """Test that default quality criteria are set."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=0.0,
            end_time=60.0,
            clip_summary="Test summary",
        )

        expected_criteria = [
            "sentence_boundaries",
            "topic_complete",
            "has_hook",
            "standalone_valid",
            "brand_relevant",
            "transcript_aligned",
        ]
        assert request.quality_criteria == expected_criteria

    def test_custom_quality_criteria(self):
        """Test custom quality criteria."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=0.0,
            end_time=60.0,
            clip_summary="Test summary",
            quality_criteria=["sentence_boundaries", "topic_complete"],
        )

        assert request.quality_criteria == ["sentence_boundaries", "topic_complete"]

    def test_duration_property(self):
        """Test duration calculation."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=10.0,
            end_time=70.0,
            clip_summary="Test summary",
        )

        assert request.duration == 60.0

    def test_brand_context(self):
        """Test brand context field."""
        brand_ctx = {
            "channel_name": "Tech Talks",
            "topics": ["AI", "Machine Learning"],
            "tone": "educational",
        }
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=0.0,
            end_time=60.0,
            clip_summary="Test summary",
            brand_context=brand_ctx,
        )

        assert request.brand_context == brand_ctx

    def test_full_transcript_context(self):
        """Test optional full transcript context."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="This is the clip segment.",
            start_time=30.0,
            end_time=60.0,
            clip_summary="Test summary",
            full_transcript_context="Before segment... This is the clip segment. ...After segment.",
        )

        assert "Before segment" in request.full_transcript_context

    def test_to_dict(self):
        """Test dictionary serialization."""
        request = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=10.0,
            end_time=70.0,
            clip_summary="Test summary",
            brand_context={"channel": "Test"},
        )

        data = request.to_dict()

        assert data["clip_id"] == "clip_01"
        assert data["start_time"] == 10.0
        assert data["end_time"] == 70.0
        assert data["duration"] == 60.0
        assert data["brand_context"] == {"channel": "Test"}
        assert "quality_criteria" in data

    def test_from_dict_roundtrip(self):
        """Test dictionary deserialization roundtrip."""
        original = ClipValidationRequest(
            clip_id="clip_01",
            transcript_segment="Test text",
            start_time=10.0,
            end_time=70.0,
            clip_summary="Test summary",
            brand_context={"channel": "Test"},
            quality_criteria=["sentence_boundaries", "topic_complete"],
        )

        data = original.to_dict()
        restored = ClipValidationRequest.from_dict(data)

        assert restored.clip_id == original.clip_id
        assert restored.transcript_segment == original.transcript_segment
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert restored.clip_summary == original.clip_summary
        assert restored.brand_context == original.brand_context
        assert restored.quality_criteria == original.quality_criteria


class TestClipValidationResponse:
    """Tests for ClipValidationResponse dataclass."""

    def test_basic_creation_valid(self):
        """Test creating a valid response."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
        )

        assert response.clip_id == "clip_01"
        assert response.is_valid is True

    def test_basic_creation_invalid(self):
        """Test creating an invalid response."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=False,
            issues=["Starts mid-sentence"],
        )

        assert response.is_valid is False
        assert response.sentence_boundaries_ok is False
        assert "Starts mid-sentence" in response.issues

    def test_default_criteria_values(self):
        """Test default criteria values (all True except has_hook)."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
        )

        assert response.sentence_boundaries_ok is True
        assert response.topic_complete is True
        assert response.has_hook is False  # Optional, default False
        assert response.standalone_valid is True
        assert response.brand_relevant is True
        assert response.transcript_aligned is True

    def test_required_criteria_passed_all_true(self):
        """Test required_criteria_passed when all required pass."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
            sentence_boundaries_ok=True,
            topic_complete=True,
            has_hook=False,  # Optional
            standalone_valid=True,
            brand_relevant=True,
            transcript_aligned=True,
        )

        assert response.required_criteria_passed is True

    def test_required_criteria_passed_one_fails(self):
        """Test required_criteria_passed when one required fails."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=True,
            topic_complete=False,  # This one fails
            has_hook=True,  # Optional, doesn't affect required
            standalone_valid=True,
            brand_relevant=True,
            transcript_aligned=True,
        )

        assert response.required_criteria_passed is False

    def test_has_hook_optional(self):
        """Test that has_hook doesn't affect required_criteria_passed."""
        response_no_hook = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
            has_hook=False,
        )

        response_with_hook = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
            has_hook=True,
        )

        # Both should pass required criteria
        assert response_no_hook.required_criteria_passed is True
        assert response_with_hook.required_criteria_passed is True

    def test_failed_criteria_none(self):
        """Test failed_criteria when all pass."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
        )

        assert response.failed_criteria == []

    def test_failed_criteria_multiple(self):
        """Test failed_criteria with multiple failures."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=False,
            topic_complete=False,
            standalone_valid=True,
            brand_relevant=False,
            transcript_aligned=True,
        )

        failed = response.failed_criteria
        assert "sentence_boundaries" in failed
        assert "topic_complete" in failed
        assert "brand_relevant" in failed
        assert "standalone_valid" not in failed
        assert "transcript_aligned" not in failed
        assert len(failed) == 3

    def test_issues_and_suggestions(self):
        """Test issues and suggestions fields."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=False,
            issues=["Starts mid-sentence", "Ends abruptly"],
            suggestions=["Extend start by 2 seconds", "Try ending at 45.5s"],
        )

        assert len(response.issues) == 2
        assert len(response.suggestions) == 2
        assert "Starts mid-sentence" in response.issues
        assert "Extend start by 2 seconds" in response.suggestions

    def test_confidence_and_tokens(self):
        """Test confidence and token tracking."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=True,
            confidence=0.95,
            tokens_used=150,
        )

        assert response.confidence == 0.95
        assert response.tokens_used == 150

    def test_to_dict(self):
        """Test dictionary serialization."""
        response = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=False,
            topic_complete=True,
            has_hook=True,
            standalone_valid=True,
            brand_relevant=False,
            transcript_aligned=True,
            issues=["Problem 1"],
            suggestions=["Fix 1"],
            confidence=0.9,
            tokens_used=100,
        )

        data = response.to_dict()

        assert data["clip_id"] == "clip_01"
        assert data["is_valid"] is False
        assert data["sentence_boundaries_ok"] is False
        assert data["has_hook"] is True
        assert data["brand_relevant"] is False
        assert data["issues"] == ["Problem 1"]
        assert data["confidence"] == 0.9
        assert data["tokens_used"] == 100
        # Computed properties
        assert data["required_criteria_passed"] is False
        assert "sentence_boundaries" in data["failed_criteria"]
        assert "brand_relevant" in data["failed_criteria"]

    def test_from_dict_roundtrip(self):
        """Test dictionary deserialization roundtrip."""
        original = ClipValidationResponse(
            clip_id="clip_01",
            is_valid=False,
            sentence_boundaries_ok=False,
            topic_complete=True,
            has_hook=True,
            standalone_valid=True,
            brand_relevant=False,
            transcript_aligned=True,
            issues=["Issue 1", "Issue 2"],
            suggestions=["Suggestion 1"],
            confidence=0.85,
            tokens_used=200,
        )

        data = original.to_dict()
        restored = ClipValidationResponse.from_dict(data)

        assert restored.clip_id == original.clip_id
        assert restored.is_valid == original.is_valid
        assert restored.sentence_boundaries_ok == original.sentence_boundaries_ok
        assert restored.topic_complete == original.topic_complete
        assert restored.has_hook == original.has_hook
        assert restored.standalone_valid == original.standalone_valid
        assert restored.brand_relevant == original.brand_relevant
        assert restored.transcript_aligned == original.transcript_aligned
        assert restored.issues == original.issues
        assert restored.suggestions == original.suggestions
        assert restored.confidence == original.confidence
        assert restored.tokens_used == original.tokens_used

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "clip_id": "clip_01",
            "is_valid": True,
        }

        response = ClipValidationResponse.from_dict(data)

        assert response.clip_id == "clip_01"
        assert response.is_valid is True
        # Check defaults
        assert response.sentence_boundaries_ok is True
        assert response.has_hook is False
        assert response.issues == []
        assert response.confidence == 0.8
