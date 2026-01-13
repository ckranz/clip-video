"""Tests for CLI review commands."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from clip_video.cli import app
from clip_video.review.queue import ReviewQueue, RejectedClip


runner = CliRunner()


class TestReviewListCommand:
    """Tests for 'clip-video review list' command."""

    def test_list_empty_queue(self, tmp_path):
        """Test listing an empty review queue."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "list"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_list_no_queue_directory(self, tmp_path):
        """Test when no review directory exists."""
        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "list"])

        assert result.exit_code == 0
        assert "No review queue found" in result.output

    def test_list_with_clips(self, tmp_path):
        """Test listing clips in the queue."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        # Add test clips
        clip1 = RejectedClip(
            clip_id="clip_10.0_70.0",
            video_path="/test/video.mp4",
            start_time=10.0,
            end_time=70.0,
            transcript_segment="Test transcript",
            rejection_reasons=["Too short", "No hook"],
        )
        clip2 = RejectedClip(
            clip_id="clip_100.0_160.0",
            video_path="/test/video.mp4",
            start_time=100.0,
            end_time=160.0,
            transcript_segment="Another test",
            rejection_reasons=["Topic incomplete"],
        )
        queue.add(clip1)
        queue.add(clip2)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "list"])

        assert result.exit_code == 0
        assert "clip_10.0_70.0" in result.output
        assert "clip_100.0_160.0" in result.output
        assert "2 clips" in result.output

    def test_list_with_limit(self, tmp_path):
        """Test limiting number of results."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        # Add multiple clips
        for i in range(5):
            clip = RejectedClip(
                clip_id=f"clip_{i}.0",
                video_path="/test/video.mp4",
                start_time=float(i * 60),
                end_time=float(i * 60 + 50),
                transcript_segment=f"Test {i}",
                rejection_reasons=["Test reason"],
            )
            queue.add(clip)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "list", "--limit", "2"])

        assert result.exit_code == 0
        assert "Showing 2 of 5" in result.output

    def test_list_filter_by_video(self, tmp_path):
        """Test filtering by video name."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip1 = RejectedClip(
            clip_id="clip_1",
            video_path="/test/video1.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Test"],
        )
        clip2 = RejectedClip(
            clip_id="clip_2",
            video_path="/test/other_video.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Test"],
        )
        queue.add(clip1)
        queue.add(clip2)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "list", "--video", "video1"])

        assert result.exit_code == 0
        assert "clip_1" in result.output


class TestReviewShowCommand:
    """Tests for 'clip-video review show' command."""

    def test_show_clip_details(self, tmp_path):
        """Test showing detailed clip information."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_10.0_70.0",
            video_path="/test/video.mp4",
            start_time=10.0,
            end_time=70.0,
            transcript_segment="This is the test transcript segment.",
            rejection_reasons=["Too short", "No hook"],
            validation_details={
                "criteria": [
                    {"criterion": "duration", "result": "pass", "reason": "OK"},
                    {"criterion": "has_hook", "result": "fail", "reason": "No hook"},
                ]
            },
        )
        queue.add(clip)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "show", "clip_10.0_70.0"])

        assert result.exit_code == 0
        assert "clip_10.0_70.0" in result.output
        assert "10.0s" in result.output
        assert "70.0s" in result.output
        assert "Too short" in result.output
        assert "No hook" in result.output
        assert "transcript" in result.output.lower()
        assert "ffplay" in result.output

    def test_show_clip_not_found(self, tmp_path):
        """Test showing a non-existent clip."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestReviewApproveCommand:
    """Tests for 'clip-video review approve' command."""

    def test_approve_clip(self, tmp_path):
        """Test approving a clip."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_10.0_70.0",
            video_path="/test/video.mp4",
            start_time=10.0,
            end_time=70.0,
            transcript_segment="Test",
            rejection_reasons=["Test reason"],
        )
        queue.add(clip)
        assert queue.count() == 1

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "approve", "clip_10.0_70.0"])

        assert result.exit_code == 0
        assert "Approved" in result.output
        assert queue.count() == 0

    def test_approve_nonexistent_clip(self, tmp_path):
        """Test approving a non-existent clip."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "approve", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestReviewRejectCommand:
    """Tests for 'clip-video review reject' command."""

    def test_reject_clip(self, tmp_path):
        """Test rejecting a clip."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_10.0_70.0",
            video_path="/test/video.mp4",
            start_time=10.0,
            end_time=70.0,
            transcript_segment="Test",
            rejection_reasons=["Test reason"],
        )
        queue.add(clip)
        assert queue.count() == 1

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "reject", "clip_10.0_70.0"])

        assert result.exit_code == 0
        assert "Rejected" in result.output
        assert queue.count() == 0


class TestReviewClearCommand:
    """Tests for 'clip-video review clear' command."""

    def test_clear_empty_queue(self, tmp_path):
        """Test clearing an empty queue."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "clear", "--yes"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_clear_with_confirmation(self, tmp_path):
        """Test clearing queue with confirmation."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        for i in range(3):
            clip = RejectedClip(
                clip_id=f"clip_{i}",
                video_path="/test/video.mp4",
                start_time=float(i * 60),
                end_time=float(i * 60 + 50),
                transcript_segment="Test",
                rejection_reasons=["Test"],
            )
            queue.add(clip)

        assert queue.count() == 3

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "clear", "--yes"])

        assert result.exit_code == 0
        assert "Cleared 3" in result.output
        assert queue.count() == 0

    def test_clear_cancelled(self, tmp_path):
        """Test cancelling clear operation."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_1",
            video_path="/test/video.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Test"],
        )
        queue.add(clip)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "clear"], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        assert queue.count() == 1  # Not cleared


class TestReviewSummaryCommand:
    """Tests for 'clip-video review summary' command."""

    def test_summary_empty_queue(self, tmp_path):
        """Test summary of empty queue."""
        review_dir = tmp_path / "review"
        review_dir.mkdir()

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "summary"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_summary_with_clips(self, tmp_path):
        """Test summary with clips in queue."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        clip1 = RejectedClip(
            clip_id="clip_1",
            video_path="/test/video1.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Too short"],
        )
        clip2 = RejectedClip(
            clip_id="clip_2",
            video_path="/test/video1.mp4",
            start_time=100.0,
            end_time=160.0,
            transcript_segment="Test",
            rejection_reasons=["No hook"],
        )
        clip3 = RejectedClip(
            clip_id="clip_3",
            video_path="/test/video2.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Too short"],
        )
        queue.add(clip1)
        queue.add(clip2)
        queue.add(clip3)

        with patch("clip_video.cli.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["review", "summary"])

        assert result.exit_code == 0
        assert "Total Clips" in result.output
        assert "3" in result.output
        assert "By Rejection Reason" in result.output
        assert "Too short" in result.output
        assert "By Source Video" in result.output


class TestReviewWithBrand:
    """Tests for review commands with brand parameter."""

    def test_list_with_brand(self, tmp_path):
        """Test listing with brand name."""
        # Create brand structure
        brand_path = tmp_path / "brands" / "test-brand"
        brand_path.mkdir(parents=True)
        (brand_path / "config.json").write_text(
            '{"name": "test-brand", "description": "", "vocabulary": {}}'
        )

        review_dir = brand_path / "review"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_1",
            video_path="/test/video.mp4",
            start_time=0.0,
            end_time=60.0,
            transcript_segment="Test",
            rejection_reasons=["Test"],
        )
        queue.add(clip)

        with patch("clip_video.cli.get_brand_path", return_value=brand_path):
            with patch("clip_video.cli.brand_exists", return_value=True):
                result = runner.invoke(app, ["review", "list", "test-brand"])

        assert result.exit_code == 0
        assert "clip_1" in result.output

    def test_list_with_invalid_brand(self, tmp_path):
        """Test listing with non-existent brand."""
        result = runner.invoke(app, ["review", "list", "nonexistent-brand"])

        assert result.exit_code == 1
        assert "does not exist" in result.output
