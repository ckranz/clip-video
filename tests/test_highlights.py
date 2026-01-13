"""Tests for highlights mode processing."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from clip_video.modes.highlights import (
    HighlightClip,
    HighlightMetadata,
    HighlightsConfig,
    HighlightsProject,
    HighlightsProcessor,
    Platform,
)
from clip_video.llm.base import HighlightSegment, HighlightAnalysis, LLMConfig
from clip_video.captions.styles import YOUTUBE_SHORTS_STYLE
from clip_video.video.portrait import YOUTUBE_SHORTS_CONFIG


class TestHighlightClip:
    """Tests for HighlightClip dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test basic clip creation."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=40.0,
            summary="Test segment",
            hook_text="Check this out!",
            reason="Great content",
            topics=["testing"],
        )

        clip = HighlightClip(
            clip_id="clip_01",
            segment=segment,
            source_video=tmp_path / "video.mp4",
        )

        assert clip.clip_id == "clip_01"
        assert clip.segment.duration == 30.0
        assert clip.created_at != ""

    def test_final_clip_path_priority(self, tmp_path):
        """Test final_clip_path returns correct priority."""
        segment = HighlightSegment(
            start_time=0, end_time=30,
            summary="", hook_text="", reason="",
        )

        clip = HighlightClip(
            clip_id="clip_01",
            segment=segment,
            source_video=tmp_path / "video.mp4",
        )

        # No paths set
        assert clip.final_clip_path is None

        # Raw only
        clip.raw_clip_path = tmp_path / "raw.mp4"
        assert clip.final_clip_path == clip.raw_clip_path

        # Portrait takes priority over raw
        clip.portrait_clip_path = tmp_path / "portrait.mp4"
        assert clip.final_clip_path == clip.portrait_clip_path

        # Captioned takes highest priority
        clip.captioned_clip_path = tmp_path / "final.mp4"
        assert clip.final_clip_path == clip.captioned_clip_path

    def test_serialization(self, tmp_path):
        """Test to_dict and from_dict."""
        segment = HighlightSegment(
            start_time=10.0,
            end_time=40.0,
            summary="Test segment",
            hook_text="Check this out!",
            reason="Great content",
            topics=["testing", "quality"],
            quality_score=0.9,
        )

        clip = HighlightClip(
            clip_id="clip_01",
            segment=segment,
            source_video=tmp_path / "video.mp4",
            raw_clip_path=tmp_path / "raw.mp4",
            metadata={"key": "value"},
        )

        data = clip.to_dict()
        loaded = HighlightClip.from_dict(data)

        assert loaded.clip_id == clip.clip_id
        assert loaded.segment.start_time == 10.0
        assert loaded.segment.end_time == 40.0
        assert loaded.metadata == {"key": "value"}


class TestHighlightMetadata:
    """Tests for HighlightMetadata dataclass."""

    def test_defaults(self):
        """Test default values."""
        metadata = HighlightMetadata(
            title="Test Title",
            hook_text="Hook!",
        )

        assert metadata.title == "Test Title"
        assert metadata.hook_text == "Hook!"
        assert metadata.description == ""
        assert metadata.hashtags == []
        assert metadata.platform_specific == {}

    def test_serialization(self):
        """Test to_dict and from_dict."""
        metadata = HighlightMetadata(
            title="Test Title",
            hook_text="Hook!",
            description="Description",
            hashtags=["#test", "#video"],
            platform_specific={
                "youtube_shorts": {"tags": ["test"]},
            },
        )

        data = metadata.to_dict()
        loaded = HighlightMetadata.from_dict(data)

        assert loaded.title == metadata.title
        assert loaded.hashtags == metadata.hashtags
        assert loaded.platform_specific == metadata.platform_specific


class TestHighlightsConfig:
    """Tests for HighlightsConfig dataclass."""

    def test_defaults(self):
        """Test default configuration with YouTube Shorts compatible durations."""
        config = HighlightsConfig()

        # Core settings
        assert config.target_clips == 5
        assert config.output_format == "mp4"
        assert Platform.YOUTUBE_SHORTS in config.platforms

        # YouTube Shorts duration constraints (30-120s)
        assert config.min_duration == 30.0
        assert config.max_duration == 120.0

        # New agentic workflow settings
        assert config.min_acceptable_clips == 3
        assert config.max_replacement_attempts == 3
        assert config.cost_ceiling_gbp == 5.0
        assert config.enable_validation_pass is True

    def test_config_duration_validation(self):
        """Test that min_duration cannot exceed max_duration."""
        with pytest.raises(ValueError, match="min_duration.*cannot be greater than.*max_duration"):
            HighlightsConfig(min_duration=60.0, max_duration=30.0)

    def test_config_min_acceptable_less_than_target(self):
        """Test that min_acceptable_clips cannot exceed target_clips."""
        with pytest.raises(ValueError, match="min_acceptable_clips.*cannot be greater than.*target_clips"):
            HighlightsConfig(target_clips=3, min_acceptable_clips=5)

    def test_config_valid_custom_values(self):
        """Test valid custom configuration values."""
        config = HighlightsConfig(
            target_clips=10,
            min_acceptable_clips=5,
            min_duration=45.0,
            max_duration=90.0,
            max_replacement_attempts=5,
            cost_ceiling_gbp=10.0,
            enable_validation_pass=False,
        )

        assert config.target_clips == 10
        assert config.min_acceptable_clips == 5
        assert config.min_duration == 45.0
        assert config.max_duration == 90.0
        assert config.max_replacement_attempts == 5
        assert config.cost_ceiling_gbp == 10.0
        assert config.enable_validation_pass is False

    def test_get_platform_config(self):
        """Test platform-specific config retrieval."""
        config = HighlightsConfig()

        yt_config = config.get_platform_config(Platform.YOUTUBE_SHORTS)
        assert yt_config.target_width == 1080
        assert yt_config.target_height == 1920

        tt_config = config.get_platform_config(Platform.TIKTOK)
        assert tt_config.video_crf == 18  # TikTok uses higher quality


class TestHighlightsProject:
    """Tests for HighlightsProject dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test basic project creation."""
        project = HighlightsProject(
            name="test_project",
            brand_name="TestBrand",
            video_path=tmp_path / "video.mp4",
        )

        assert project.name == "test_project"
        assert project.brand_name == "TestBrand"
        assert project.created_at != ""

    def test_directory_properties(self, tmp_path):
        """Test directory path properties."""
        project = HighlightsProject(
            name="test_project",
            brand_name="TestBrand",
            video_path=tmp_path / "video.mp4",
        )
        project.project_root = tmp_path / "project"

        assert project.clips_dir == tmp_path / "project" / "clips"
        assert project.raw_clips_dir == tmp_path / "project" / "clips" / "raw"
        assert project.portrait_clips_dir == tmp_path / "project" / "clips" / "portrait"
        assert project.final_clips_dir == tmp_path / "project" / "clips" / "final"
        assert project.metadata_dir == tmp_path / "project" / "metadata"
        assert project.state_file == tmp_path / "project" / "project_state.json"

    def test_summary(self, tmp_path):
        """Test get_summary method."""
        project = HighlightsProject(
            name="test_project",
            brand_name="TestBrand",
            video_path=tmp_path / "video.mp4",
        )

        summary = project.get_summary()

        assert summary["project_name"] == "test_project"
        assert summary["brand_name"] == "TestBrand"
        assert summary["has_transcript"] is False
        assert summary["has_analysis"] is False
        assert summary["clips_generated"] == 0

    def test_serialization(self, tmp_path):
        """Test to_dict and from_dict."""
        project = HighlightsProject(
            name="test_project",
            brand_name="TestBrand",
            video_path=tmp_path / "video.mp4",
            transcript_text="Hello world",
        )

        data = project.to_dict()
        loaded = HighlightsProject.from_dict(data)

        assert loaded.name == project.name
        assert loaded.brand_name == project.brand_name
        assert loaded.transcript_text == "Hello world"

    def test_save_and_load(self, tmp_path):
        """Test project persistence."""
        project = HighlightsProject(
            name="test_project",
            brand_name="TestBrand",
            video_path=tmp_path / "video.mp4",
        )
        project.project_root = tmp_path / "project"
        project.transcript_text = "Test transcript"

        project.save()

        assert project.state_file.exists()

        loaded = HighlightsProject.load(project.state_file)

        assert loaded.name == project.name
        assert loaded.transcript_text == project.transcript_text


class TestHighlightsProcessor:
    """Tests for HighlightsProcessor class."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()

            assert processor.config is not None
            assert processor.config.target_clips == 5

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = HighlightsConfig(target_clips=10)

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor(config=config)

            assert processor.config.target_clips == 10

    def test_create_project(self, tmp_path):
        """Test project creation."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            project = processor.create_project(
                name="test",
                brand_name="TestBrand",
                video_path=video_path,
                project_root=tmp_path / "project",
            )

            assert project.name == "test"
            assert project.raw_clips_dir.exists()
            assert project.portrait_clips_dir.exists()
            assert project.final_clips_dir.exists()
            assert project.metadata_dir.exists()

    def test_load_description(self, tmp_path):
        """Test loading session description."""
        desc_path = tmp_path / "description.txt"
        desc_path.write_text("This is a test session about Kubernetes.")

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
                description_path=desc_path,
            )

            description = processor.load_description(project)

            assert "Kubernetes" in description

    def test_transcribe_with_segments(self, tmp_path):
        """Test transcription with provided segments."""
        from clip_video.transcription import TranscriptionSegment

        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Hello world"),
            TranscriptionSegment(start=5.0, end=10.0, text="Testing"),
        ]

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
            )
            project.project_root = tmp_path / "project"
            project.project_root.mkdir(parents=True)

            transcript = processor.transcribe(project, segments)

            assert "Hello world" in transcript
            assert "[0.0s - 5.0s]" in transcript

    def test_transcribe_existing(self, tmp_path):
        """Test that existing transcript is returned."""
        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
                transcript_text="Existing transcript",
            )

            transcript = processor.transcribe(project)

            assert transcript == "Existing transcript"

    def test_analyze(self, tmp_path):
        """Test highlight analysis."""
        mock_llm = Mock()
        mock_analysis = HighlightAnalysis(
            video_id="test",
            segments=[
                HighlightSegment(
                    start_time=10.0,
                    end_time=40.0,
                    summary="Great segment",
                    hook_text="Check this out!",
                    reason="High engagement",
                    topics=["testing"],
                ),
            ],
        )
        mock_llm.analyze_transcript.return_value = mock_analysis

        with patch("clip_video.modes.highlights.ClaudeLLM", return_value=mock_llm):
            processor = HighlightsProcessor()
            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
                transcript_text="Test transcript content",
            )
            project.project_root = tmp_path / "project"
            project.project_root.mkdir(parents=True)

            analysis = processor.analyze(project)

            assert analysis is not None
            assert len(analysis.segments) == 1
            assert analysis.segments[0].summary == "Great segment"

    def test_analyze_filters_by_duration(self, tmp_path):
        """Test that analysis filters segments by duration (30-120s for YouTube Shorts)."""
        mock_llm = Mock()
        mock_analysis = HighlightAnalysis(
            video_id="test",
            segments=[
                HighlightSegment(
                    start_time=0.0,
                    end_time=20.0,  # Too short (20s < 30s min)
                    summary="Short",
                    hook_text="",
                    reason="",
                ),
                HighlightSegment(
                    start_time=0.0,
                    end_time=60.0,  # Valid (60s within 30-120s)
                    summary="Valid",
                    hook_text="",
                    reason="",
                ),
                HighlightSegment(
                    start_time=0.0,
                    end_time=150.0,  # Too long (150s > 120s max)
                    summary="Long",
                    hook_text="",
                    reason="",
                ),
            ],
        )
        mock_llm.analyze_transcript.return_value = mock_analysis

        with patch("clip_video.modes.highlights.ClaudeLLM", return_value=mock_llm):
            processor = HighlightsProcessor()
            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
                transcript_text="Test transcript",
            )
            project.project_root = tmp_path / "project"
            project.project_root.mkdir(parents=True)

            analysis = processor.analyze(project)

            # Only the valid-duration segment should remain
            assert len(analysis.segments) == 1
            assert analysis.segments[0].summary == "Valid"

    @patch("clip_video.modes.highlights.FFmpegWrapper")
    def test_extract_clips(self, mock_ffmpeg_class, tmp_path):
        """Test clip extraction."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            processor.ffmpeg = mock_ffmpeg

            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=video_path,
            )
            project.project_root = tmp_path / "project"
            project.raw_clips_dir.mkdir(parents=True)
            project.analysis = HighlightAnalysis(
                video_id="test",
                segments=[
                    HighlightSegment(
                        start_time=10.0,
                        end_time=40.0,
                        summary="Test",
                        hook_text="",
                        reason="",
                    ),
                ],
            )

            clips = processor.extract_clips(project)

            assert len(clips) == 1
            assert clips[0].clip_id == "clip_01"
            mock_ffmpeg.extract_clip.assert_called_once()

    @patch("clip_video.modes.highlights.PortraitConverter")
    def test_convert_to_portrait(self, mock_converter_class, tmp_path):
        """Test portrait conversion."""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter

        raw_clip = tmp_path / "raw.mp4"
        raw_clip.touch()

        # Mock ffmpeg to return valid video info
        from clip_video.ffmpeg import VideoInfo
        mock_ffmpeg = Mock()
        mock_ffmpeg.get_video_info.return_value = VideoInfo(
            width=1920,
            height=1080,
            duration=30.0,
            fps=30.0,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
        )

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()
            processor.portrait_converter = mock_converter
            processor.ffmpeg = mock_ffmpeg

            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
            )
            project.project_root = tmp_path / "project"
            project.portrait_clips_dir.mkdir(parents=True)

            segment = HighlightSegment(
                start_time=0.0, end_time=30.0,
                summary="", hook_text="", reason="",
            )
            project.clips = [
                HighlightClip(
                    clip_id="clip_01",
                    segment=segment,
                    source_video=tmp_path / "video.mp4",
                    raw_clip_path=raw_clip,
                ),
            ]

            clips = processor.convert_to_portrait(project)

            assert len(clips) == 1
            mock_converter.convert.assert_called_once()

    def test_generate_metadata(self, tmp_path):
        """Test metadata generation."""
        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor()

            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
            )
            project.project_root = tmp_path / "project"
            project.metadata_dir.mkdir(parents=True)

            segment = HighlightSegment(
                start_time=10.0,
                end_time=40.0,
                summary="Great highlight",
                hook_text="You won't believe this!",
                reason="High engagement content",
                topics=["testing", "quality"],
                quality_score=0.95,
            )
            project.clips = [
                HighlightClip(
                    clip_id="clip_01",
                    segment=segment,
                    source_video=tmp_path / "video.mp4",
                ),
            ]

            metadata_list = processor.generate_metadata(project)

            assert len(metadata_list) == 1
            assert metadata_list[0]["title"] == "Great highlight"
            assert metadata_list[0]["hook_text"] == "You won't believe this!"
            assert "#testing" in metadata_list[0]["hashtags"]

            # Check metadata file was written
            metadata_file = project.metadata_dir / "clip_01_metadata.json"
            assert metadata_file.exists()

    def test_get_cost_estimate(self, tmp_path):
        """Test cost estimation."""
        mock_llm = Mock()
        mock_llm.estimate_cost.return_value = 0.05

        with patch("clip_video.modes.highlights.ClaudeLLM", return_value=mock_llm):
            processor = HighlightsProcessor()

            cost = processor.get_cost_estimate("Test transcript")

            assert cost == 0.05
            mock_llm.estimate_cost.assert_called_once_with("Test transcript")


class TestPlatform:
    """Tests for Platform constants."""

    def test_platform_values(self):
        """Test platform constant values."""
        assert Platform.YOUTUBE_SHORTS == "youtube_shorts"
        assert Platform.TIKTOK == "tiktok"
        assert Platform.INSTAGRAM_REELS == "instagram_reels"


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called(self, tmp_path):
        """Test that progress callback is invoked."""
        progress_calls = []

        def callback(stage: str, progress: float):
            progress_calls.append((stage, progress))

        with patch("clip_video.modes.highlights.ClaudeLLM"):
            processor = HighlightsProcessor(progress_callback=callback)

            project = HighlightsProject(
                name="test",
                brand_name="TestBrand",
                video_path=tmp_path / "video.mp4",
                transcript_text="Existing transcript",
            )

            processor.transcribe(project)

            # Should have been called for transcription stage
            assert any(stage == "transcription" for stage, _ in progress_calls)
