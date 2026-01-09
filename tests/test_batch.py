"""Tests for batch processing orchestration."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from clip_video.batch import (
    VideoStatus,
    VideoResult,
    BatchConfig,
    BatchJob,
    BatchProcessor,
)


class TestVideoStatus:
    """Tests for VideoStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert VideoStatus.PENDING.value == "pending"
        assert VideoStatus.IN_PROGRESS.value == "in_progress"
        assert VideoStatus.COMPLETED.value == "completed"
        assert VideoStatus.FAILED.value == "failed"
        assert VideoStatus.SKIPPED.value == "skipped"


class TestVideoResult:
    """Tests for VideoResult dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test basic result creation."""
        result = VideoResult(video_path=tmp_path / "video.mp4")

        assert result.status == VideoStatus.PENDING
        assert result.clips_generated == 0
        assert result.error_message == ""

    def test_duration_calculation(self, tmp_path):
        """Test duration calculation."""
        result = VideoResult(
            video_path=tmp_path / "video.mp4",
            started_at="2025-01-09T10:00:00",
            completed_at="2025-01-09T10:05:30",
        )

        assert result.duration == 330.0  # 5 minutes 30 seconds

    def test_duration_none_when_incomplete(self, tmp_path):
        """Test duration is None when times not set."""
        result = VideoResult(video_path=tmp_path / "video.mp4")

        assert result.duration is None

    def test_serialization(self, tmp_path):
        """Test to_dict and from_dict."""
        result = VideoResult(
            video_path=tmp_path / "video.mp4",
            status=VideoStatus.COMPLETED,
            project_name="test_project",
            clips_generated=3,
            clip_paths=["clip1.mp4", "clip2.mp4", "clip3.mp4"],
        )

        data = result.to_dict()
        loaded = VideoResult.from_dict(data)

        assert loaded.status == VideoStatus.COMPLETED
        assert loaded.project_name == "test_project"
        assert loaded.clips_generated == 3
        assert len(loaded.clip_paths) == 3


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_defaults(self):
        """Test default configuration."""
        config = BatchConfig()

        assert config.max_parallel == 4
        assert config.continue_on_error is True
        assert config.skip_completed is True
        assert config.highlights_config is not None


class TestBatchJob:
    """Tests for BatchJob dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test basic job creation."""
        videos = [tmp_path / "video1.mp4", tmp_path / "video2.mp4"]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        assert job.name == "test_batch"
        assert job.total_videos == 2
        assert job.pending_count == 2
        assert job.completed_count == 0

    def test_results_initialized(self, tmp_path):
        """Test that results are initialized for all videos."""
        videos = [tmp_path / "video1.mp4", tmp_path / "video2.mp4"]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        assert len(job.results) == 2
        for video in videos:
            assert str(video) in job.results
            assert job.results[str(video)].status == VideoStatus.PENDING

    def test_progress_calculation(self, tmp_path):
        """Test progress percentage calculation."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(4)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        # Initially 0%
        assert job.progress_percent == 0.0

        # Complete one
        job.results[str(videos[0])].status = VideoStatus.COMPLETED
        assert job.progress_percent == 25.0

        # Fail one
        job.results[str(videos[1])].status = VideoStatus.FAILED
        assert job.progress_percent == 50.0

    def test_get_pending_videos(self, tmp_path):
        """Test getting pending video list."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        # Complete first video
        job.results[str(videos[0])].status = VideoStatus.COMPLETED

        pending = job.get_pending_videos()
        assert len(pending) == 2
        assert videos[0] not in pending
        assert videos[1] in pending
        assert videos[2] in pending

    def test_get_failed_videos(self, tmp_path):
        """Test getting failed video list."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        # Fail one video
        job.results[str(videos[1])].status = VideoStatus.FAILED
        job.results[str(videos[1])].error_message = "Test error"

        failed = job.get_failed_videos()
        assert len(failed) == 1
        assert failed[0][0] == videos[1]
        assert failed[0][1] == "Test error"

    def test_total_clips_generated(self, tmp_path):
        """Test total clips calculation."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        job.results[str(videos[0])].clips_generated = 3
        job.results[str(videos[1])].clips_generated = 5
        job.results[str(videos[2])].clips_generated = 2

        assert job.total_clips_generated == 10

    def test_get_summary(self, tmp_path):
        """Test summary generation."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(4)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )

        job.results[str(videos[0])].status = VideoStatus.COMPLETED
        job.results[str(videos[0])].clips_generated = 3
        job.results[str(videos[1])].status = VideoStatus.FAILED

        summary = job.get_summary()

        assert summary["job_name"] == "test_batch"
        assert summary["total_videos"] == 4
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["pending"] == 2
        assert summary["total_clips_generated"] == 3

    def test_serialization(self, tmp_path):
        """Test to_dict and from_dict."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(2)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )
        job.results[str(videos[0])].status = VideoStatus.COMPLETED
        job.results[str(videos[0])].clips_generated = 5

        data = job.to_dict()
        loaded = BatchJob.from_dict(data)

        assert loaded.name == "test_batch"
        assert loaded.brand_name == "TestBrand"
        assert loaded.total_videos == 2
        assert loaded.completed_count == 1

    def test_save_and_load(self, tmp_path):
        """Test job persistence."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(2)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )
        job.job_root = tmp_path / "batch"
        job.results[str(videos[0])].status = VideoStatus.COMPLETED

        job.save()

        assert job.state_file.exists()

        loaded = BatchJob.load(job.state_file)

        assert loaded.name == job.name
        assert loaded.completed_count == 1

    def test_generate_report(self, tmp_path):
        """Test report generation."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        job = BatchJob(
            name="test_batch",
            brand_name="TestBrand",
            video_paths=videos,
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        job.results[str(videos[0])].status = VideoStatus.COMPLETED
        job.results[str(videos[0])].clips_generated = 3
        job.results[str(videos[0])].clip_paths = ["clip1.mp4", "clip2.mp4"]
        job.results[str(videos[1])].status = VideoStatus.FAILED
        job.results[str(videos[1])].error_message = "Test error"

        report = job.generate_report()

        assert "summary" in report
        assert "videos" in report
        assert len(report["videos"]["completed"]) == 1
        assert len(report["videos"]["failed"]) == 1
        assert len(report["videos"]["pending"]) == 1
        assert len(report["all_clips"]) == 2

        # Check report file was saved
        assert job.report_file.exists()


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        processor = BatchProcessor()

        assert processor.config is not None
        assert processor.config.max_parallel == 4

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = BatchConfig(max_parallel=8)
        processor = BatchProcessor(config=config)

        assert processor.config.max_parallel == 8

    def test_create_job(self, tmp_path):
        """Test job creation."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(2)]
        for v in videos:
            v.touch()

        processor = BatchProcessor()
        job = processor.create_job(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
            job_root=tmp_path / "batch",
        )

        assert job.name == "test"
        assert job.total_videos == 2
        assert job.state_file.exists()

    def test_load_video_list_text_file(self, tmp_path):
        """Test loading videos from text file."""
        list_file = tmp_path / "videos.txt"
        list_file.write_text(
            "/path/to/video1.mp4\n"
            "/path/to/video2.mp4\n"
            "# This is a comment\n"
            "/path/to/video3.mp4\n"
        )

        processor = BatchProcessor()
        paths = processor.load_video_list(list_file)

        assert len(paths) == 3
        assert Path("/path/to/video1.mp4") in paths
        assert Path("/path/to/video2.mp4") in paths
        assert Path("/path/to/video3.mp4") in paths

    def test_load_video_list_json_file(self, tmp_path):
        """Test loading videos from JSON file."""
        list_file = tmp_path / "videos.json"
        list_file.write_text(json.dumps([
            "/path/to/video1.mp4",
            "/path/to/video2.mp4",
        ]))

        processor = BatchProcessor()
        paths = processor.load_video_list(list_file)

        assert len(paths) == 2

    def test_load_video_list_directory(self, tmp_path):
        """Test loading videos from directory."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "video1.mp4").touch()
        (video_dir / "video2.mkv").touch()
        (video_dir / "readme.txt").touch()  # Should be ignored

        processor = BatchProcessor()
        paths = processor.load_video_list(video_dir)

        assert len(paths) == 2
        assert any(p.suffix == ".mp4" for p in paths)
        assert any(p.suffix == ".mkv" for p in paths)

    @patch("clip_video.batch.HighlightsProcessor")
    def test_process_single_video_success(self, mock_processor_class, tmp_path):
        """Test successful single video processing."""
        mock_processor = Mock()
        mock_project = Mock()
        mock_project.clips = [Mock(), Mock(), Mock()]
        for i, clip in enumerate(mock_project.clips):
            clip.final_clip_path = tmp_path / f"clip{i}.mp4"

        mock_processor.create_project.return_value = mock_project
        mock_processor.process.return_value = mock_project
        mock_processor_class.return_value = mock_processor

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=[video_path],
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        processor = BatchProcessor()
        result = processor._process_single_video(job, video_path)

        assert result.status == VideoStatus.COMPLETED
        assert result.clips_generated == 3

    @patch("clip_video.batch.HighlightsProcessor")
    def test_process_single_video_failure(self, mock_processor_class, tmp_path):
        """Test failed single video processing."""
        mock_processor = Mock()
        mock_processor.create_project.side_effect = ValueError("Test error")
        mock_processor_class.return_value = mock_processor

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        config = BatchConfig(continue_on_error=True)
        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=[video_path],
            config=config,
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        processor = BatchProcessor(config=config)
        result = processor._process_single_video(job, video_path)

        assert result.status == VideoStatus.FAILED
        assert "Test error" in result.error_message

    @patch("clip_video.batch.HighlightsProcessor")
    def test_process_sequential(self, mock_processor_class, tmp_path):
        """Test sequential processing."""
        mock_processor = Mock()
        mock_project = Mock()
        mock_project.clips = []
        mock_processor.create_project.return_value = mock_project
        mock_processor.process.return_value = mock_project
        mock_processor_class.return_value = mock_processor

        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        for v in videos:
            v.touch()

        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        processor = BatchProcessor()
        result = processor.process_sequential(job)

        assert result.completed_count == 3
        assert result.pending_count == 0

    @patch("clip_video.batch.HighlightsProcessor")
    def test_process_parallel(self, mock_processor_class, tmp_path):
        """Test parallel processing."""
        mock_processor = Mock()
        mock_project = Mock()
        mock_project.clips = []
        mock_processor.create_project.return_value = mock_project
        mock_processor.process.return_value = mock_project
        mock_processor_class.return_value = mock_processor

        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        for v in videos:
            v.touch()

        config = BatchConfig(max_parallel=2)
        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
            config=config,
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        processor = BatchProcessor(config=config)
        result = processor.process_parallel(job)

        assert result.completed_count == 3

    @patch("clip_video.batch.HighlightsProcessor")
    def test_resume_job(self, mock_processor_class, tmp_path):
        """Test resuming an interrupted job."""
        mock_processor = Mock()
        mock_project = Mock()
        mock_project.clips = []
        mock_processor.create_project.return_value = mock_project
        mock_processor.process.return_value = mock_project
        mock_processor_class.return_value = mock_processor

        videos = [tmp_path / f"video{i}.mp4" for i in range(3)]
        for v in videos:
            v.touch()

        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
        )
        job.job_root = tmp_path / "batch"
        job.job_root.mkdir(parents=True)

        # Mark first video as already completed
        job.results[str(videos[0])].status = VideoStatus.COMPLETED

        processor = BatchProcessor()
        result = processor.resume(job)

        # Should only process the remaining 2
        assert result.completed_count == 3  # All completed now
        assert mock_processor.create_project.call_count == 2  # Only called for pending

    def test_get_batch_status(self, tmp_path):
        """Test batch status retrieval."""
        videos = [tmp_path / f"video{i}.mp4" for i in range(4)]
        job = BatchJob(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
        )

        job.results[str(videos[0])].status = VideoStatus.COMPLETED
        job.results[str(videos[0])].clips_generated = 3
        job.results[str(videos[1])].status = VideoStatus.FAILED
        job.results[str(videos[1])].error_message = "Error"

        processor = BatchProcessor()
        status = processor.get_batch_status(job)

        assert status["job_name"] == "test"
        assert status["progress"]["total"] == 4
        assert status["progress"]["completed"] == 1
        assert status["progress"]["failed"] == 1
        assert status["progress"]["pending"] == 2
        assert status["clips_generated"] == 3
        assert len(status["failed_videos"]) == 1


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called(self, tmp_path):
        """Test that progress callback is invoked."""
        progress_calls = []

        def callback(stage: str, current: int, total: int, message: str):
            progress_calls.append((stage, current, total, message))

        videos = [tmp_path / f"video{i}.mp4" for i in range(2)]
        for v in videos:
            v.touch()

        processor = BatchProcessor(progress_callback=callback)
        job = processor.create_job(
            name="test",
            brand_name="TestBrand",
            video_paths=videos,
            job_root=tmp_path / "batch",
        )

        # Manually trigger progress
        processor._report_progress("test", 1, 2, "Testing")

        assert len(progress_calls) == 1
        assert progress_calls[0] == ("test", 1, 2, "Testing")
