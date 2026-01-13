"""Tests for review queue system."""

import pytest
import json
from pathlib import Path

from clip_video.review.queue import RejectedClip, ReviewQueue


class TestRejectedClip:
    """Tests for RejectedClip dataclass."""

    def test_basic_creation(self):
        """Test basic clip creation."""
        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="This is the transcript text.",
            rejection_reasons=["Too short", "Overlaps with approved clip"],
        )

        assert clip.clip_id == "clip_01"
        assert clip.video_path == "/path/to/video.mp4"
        assert clip.start_time == 10.0
        assert clip.end_time == 40.0
        assert clip.rejected_at != ""  # Auto-set

    def test_duration_property(self):
        """Test duration calculation."""
        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=70.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        assert clip.duration == 60.0

    def test_preview_command(self):
        """Test preview command generation."""
        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        cmd = clip.preview_command
        assert "clip-video preview" in cmd
        assert "/path/to/video.mp4" in cmd
        assert "--start 10.0" in cmd
        assert "--end 40.0" in cmd

    def test_ffplay_command(self):
        """Test ffplay command generation."""
        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        cmd = clip.ffplay_command
        assert "ffplay" in cmd
        assert "-ss 10.0" in cmd
        assert "-t 30.0" in cmd  # duration

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict maintain data."""
        original = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="This is the transcript.",
            rejection_reasons=["Duration too short", "Overlaps"],
            validation_details={"duration": 30.0, "overlap": True},
            replacement_attempts=2,
        )

        data = original.to_dict()

        # Check serialized data
        assert data["clip_id"] == "clip_01"
        assert data["duration"] == 30.0
        assert data["preview_command"] is not None
        assert "clip-video preview" in data["preview_command"]

        # Roundtrip
        loaded = RejectedClip.from_dict(data)

        assert loaded.clip_id == original.clip_id
        assert loaded.video_path == original.video_path
        assert loaded.start_time == original.start_time
        assert loaded.rejection_reasons == original.rejection_reasons
        assert loaded.validation_details == original.validation_details
        assert loaded.replacement_attempts == 2


class TestReviewQueueAdd:
    """Tests for adding clips to the queue."""

    def test_add_clip(self, tmp_path):
        """Test adding a clip to the queue."""
        queue = ReviewQueue(tmp_path / "review")

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        filepath = queue.add(clip)

        assert filepath.exists()
        assert filepath.suffix == ".json"
        assert "clip_01" in filepath.name

    def test_add_creates_directory(self, tmp_path):
        """Test that add creates review directory if needed."""
        review_dir = tmp_path / "new_review_dir"
        queue = ReviewQueue(review_dir)

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        queue.add(clip)

        assert review_dir.exists()

    def test_add_file_content(self, tmp_path):
        """Test that added file contains correct JSON."""
        queue = ReviewQueue(tmp_path / "review")

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/path/to/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Test transcript",
            rejection_reasons=["Duration too short"],
        )

        filepath = queue.add(clip)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["clip_id"] == "clip_01"
        assert data["transcript_segment"] == "Test transcript"
        assert "preview_command" in data


class TestReviewQueueList:
    """Tests for listing clips in the queue."""

    def test_list_all_empty(self, tmp_path):
        """Test listing empty queue."""
        queue = ReviewQueue(tmp_path / "review")

        clips = queue.list_all()

        assert clips == []

    def test_list_all(self, tmp_path):
        """Test listing all clips."""
        queue = ReviewQueue(tmp_path / "review")

        # Add multiple clips
        for i in range(3):
            clip = RejectedClip(
                clip_id=f"clip_{i:02d}",
                video_path="/path/to/video.mp4",
                start_time=float(i * 30),
                end_time=float(i * 30 + 30),
                transcript_segment=f"Text {i}",
                rejection_reasons=["Test"],
            )
            queue.add(clip)

        clips = queue.list_all()

        assert len(clips) == 3

    def test_list_sorted_by_time(self, tmp_path):
        """Test that list is sorted by rejection time (newest first)."""
        queue = ReviewQueue(tmp_path / "review")

        # Add clips with explicit timestamps
        clip1 = RejectedClip(
            clip_id="clip_01",
            video_path="/video.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="First",
            rejection_reasons=["Test"],
            rejected_at="2024-01-01T10:00:00",
        )
        clip2 = RejectedClip(
            clip_id="clip_02",
            video_path="/video.mp4",
            start_time=30.0,
            end_time=60.0,
            transcript_segment="Second",
            rejection_reasons=["Test"],
            rejected_at="2024-01-01T12:00:00",  # Later
        )

        queue.add(clip1)
        queue.add(clip2)

        clips = queue.list_all()

        # clip_02 should be first (newer)
        assert clips[0].clip_id == "clip_02"
        assert clips[1].clip_id == "clip_01"


class TestReviewQueueGet:
    """Tests for getting specific clips."""

    def test_get_existing(self, tmp_path):
        """Test getting an existing clip."""
        queue = ReviewQueue(tmp_path / "review")

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )
        queue.add(clip)

        found = queue.get("clip_01")

        assert found is not None
        assert found.clip_id == "clip_01"

    def test_get_nonexistent(self, tmp_path):
        """Test getting a nonexistent clip."""
        queue = ReviewQueue(tmp_path / "review")

        found = queue.get("nonexistent")

        assert found is None


class TestReviewQueueRemove:
    """Tests for removing clips."""

    def test_remove_existing(self, tmp_path):
        """Test removing an existing clip."""
        queue = ReviewQueue(tmp_path / "review")

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )
        queue.add(clip)

        result = queue.remove("clip_01")

        assert result is True
        assert queue.get("clip_01") is None

    def test_remove_nonexistent(self, tmp_path):
        """Test removing a nonexistent clip."""
        queue = ReviewQueue(tmp_path / "review")

        result = queue.remove("nonexistent")

        assert result is False


class TestReviewQueueSummary:
    """Tests for queue summary."""

    def test_get_summary_empty(self, tmp_path):
        """Test summary of empty queue."""
        queue = ReviewQueue(tmp_path / "review")

        summary = queue.get_summary()

        assert summary["total_clips"] == 0
        assert summary["by_reason"] == {}
        assert summary["total_duration"] == 0

    def test_get_summary(self, tmp_path):
        """Test summary with clips."""
        queue = ReviewQueue(tmp_path / "review")

        # Add clips with different reasons
        clip1 = RejectedClip(
            clip_id="clip_01",
            video_path="/video1.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="Text",
            rejection_reasons=["Too short: 20s < 30s"],
        )
        clip2 = RejectedClip(
            clip_id="clip_02",
            video_path="/video1.mp4",
            start_time=30.0,
            end_time=60.0,
            transcript_segment="Text",
            rejection_reasons=["Overlaps with segment"],
        )
        clip3 = RejectedClip(
            clip_id="clip_03",
            video_path="/video2.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="Text",
            rejection_reasons=["Too short: 25s < 30s"],
        )

        queue.add(clip1)
        queue.add(clip2)
        queue.add(clip3)

        summary = queue.get_summary()

        assert summary["total_clips"] == 3
        assert summary["by_reason"]["Too short"] == 2  # Grouped by prefix
        assert summary["by_reason"]["Overlaps with segment"] == 1
        assert summary["total_duration"] == 90.0

    def test_count(self, tmp_path):
        """Test count method."""
        queue = ReviewQueue(tmp_path / "review")

        assert queue.count() == 0

        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/video.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )
        queue.add(clip)

        assert queue.count() == 1


class TestReviewQueueClear:
    """Tests for clearing the queue."""

    def test_clear(self, tmp_path):
        """Test clearing all clips."""
        queue = ReviewQueue(tmp_path / "review")

        # Add clips
        for i in range(5):
            clip = RejectedClip(
                clip_id=f"clip_{i:02d}",
                video_path="/video.mp4",
                start_time=float(i * 30),
                end_time=float(i * 30 + 30),
                transcript_segment="Text",
                rejection_reasons=["Test"],
            )
            queue.add(clip)

        assert queue.count() == 5

        removed = queue.clear()

        assert removed == 5
        assert queue.count() == 0


class TestReviewQueueListByVideo:
    """Tests for filtering by video."""

    def test_list_by_video(self, tmp_path):
        """Test listing clips from specific video."""
        queue = ReviewQueue(tmp_path / "review")

        # Add clips from different videos
        clip1 = RejectedClip(
            clip_id="clip_01",
            video_path="/video1.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )
        clip2 = RejectedClip(
            clip_id="clip_02",
            video_path="/video2.mp4",
            start_time=0.0,
            end_time=30.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )
        clip3 = RejectedClip(
            clip_id="clip_03",
            video_path="/video1.mp4",
            start_time=30.0,
            end_time=60.0,
            transcript_segment="Text",
            rejection_reasons=["Test"],
        )

        queue.add(clip1)
        queue.add(clip2)
        queue.add(clip3)

        video1_clips = queue.list_by_video("/video1.mp4")

        assert len(video1_clips) == 2
        assert all(c.video_path == "/video1.mp4" for c in video1_clips)


class TestReviewQueueFilesystem:
    """Integration tests for filesystem operations."""

    def test_review_queue_filesystem_operations(self, tmp_path):
        """Test full filesystem workflow."""
        review_dir = tmp_path / "review"
        queue = ReviewQueue(review_dir)

        # Add clip
        clip = RejectedClip(
            clip_id="clip_01",
            video_path="/video.mp4",
            start_time=10.0,
            end_time=40.0,
            transcript_segment="Test transcript segment with full context.",
            rejection_reasons=["Duration too short", "Overlaps"],
            validation_details={"duration": {"actual": 30, "min": 30}},
        )

        filepath = queue.add(clip)

        # Verify file exists and is readable
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify human-readable content
        assert "transcript_segment" in data
        assert "preview_command" in data
        assert "rejection_reasons" in data

        # Reload queue and verify persistence
        queue2 = ReviewQueue(review_dir)
        loaded = queue2.get("clip_01")

        assert loaded is not None
        assert loaded.transcript_segment == "Test transcript segment with full context."
        assert loaded.rejection_reasons == ["Duration too short", "Overlaps"]
