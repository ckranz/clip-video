"""Tests for portrait video conversion."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from clip_video.video.portrait import (
    AspectRatio,
    PortraitConfig,
    PortraitConverter,
    calculate_crop_region,
    YOUTUBE_SHORTS_CONFIG,
    TIKTOK_CONFIG,
    INSTAGRAM_REELS_CONFIG,
    INSTAGRAM_FEED_CONFIG,
    LINKEDIN_CONFIG,
)


class TestAspectRatio:
    """Tests for AspectRatio enum."""

    def test_portrait_9_16(self):
        """Test 9:16 portrait ratio."""
        ratio = AspectRatio.PORTRAIT_9_16
        assert ratio.value == "9:16"
        assert ratio.width_ratio == 9 / 16
        assert ratio.is_portrait is True

    def test_portrait_4_5(self):
        """Test 4:5 portrait ratio."""
        ratio = AspectRatio.PORTRAIT_4_5
        assert ratio.value == "4:5"
        assert ratio.width_ratio == 4 / 5
        assert ratio.is_portrait is True

    def test_square_1_1(self):
        """Test 1:1 square ratio."""
        ratio = AspectRatio.SQUARE_1_1
        assert ratio.value == "1:1"
        assert ratio.width_ratio == 1.0
        assert ratio.is_portrait is False

    def test_landscape_16_9(self):
        """Test 16:9 landscape ratio."""
        ratio = AspectRatio.LANDSCAPE_16_9
        assert ratio.value == "16:9"
        assert ratio.width_ratio == 16 / 9
        assert ratio.is_portrait is False

    def test_landscape_4_3(self):
        """Test 4:3 landscape ratio."""
        ratio = AspectRatio.LANDSCAPE_4_3
        assert ratio.value == "4:3"
        assert ratio.width_ratio == 4 / 3
        assert ratio.is_portrait is False


class TestPortraitConfig:
    """Tests for PortraitConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = PortraitConfig()

        assert config.target_ratio == AspectRatio.PORTRAIT_9_16
        assert config.target_width == 1080
        assert config.crop_x_offset == 0.5
        assert config.crop_y_offset == 0.5
        assert config.video_codec == "libx264"
        assert config.video_crf == 23
        assert config.video_preset == "medium"
        assert config.audio_codec == "aac"
        assert config.audio_bitrate == "192k"
        assert config.faststart is True

    def test_target_height_9_16(self):
        """Test height calculation for 9:16 ratio."""
        config = PortraitConfig(
            target_ratio=AspectRatio.PORTRAIT_9_16,
            target_width=1080,
        )
        assert config.target_height == 1920

    def test_target_height_4_5(self):
        """Test height calculation for 4:5 ratio."""
        config = PortraitConfig(
            target_ratio=AspectRatio.PORTRAIT_4_5,
            target_width=1080,
        )
        assert config.target_height == 1350

    def test_target_height_1_1(self):
        """Test height calculation for 1:1 ratio."""
        config = PortraitConfig(
            target_ratio=AspectRatio.SQUARE_1_1,
            target_width=1080,
        )
        assert config.target_height == 1080

    def test_dimensions(self):
        """Test dimensions property."""
        config = PortraitConfig(
            target_ratio=AspectRatio.PORTRAIT_9_16,
            target_width=1080,
        )
        assert config.dimensions == (1080, 1920)

    def test_custom_config(self):
        """Test custom configuration."""
        config = PortraitConfig(
            target_ratio=AspectRatio.PORTRAIT_4_5,
            target_width=720,
            crop_x_offset=0.3,
            video_crf=18,
            video_preset="slow",
        )

        assert config.target_ratio == AspectRatio.PORTRAIT_4_5
        assert config.target_width == 720
        assert config.crop_x_offset == 0.3
        assert config.video_crf == 18
        assert config.video_preset == "slow"


class TestCalculateCropRegion:
    """Tests for calculate_crop_region function."""

    def test_landscape_to_portrait_center_crop(self):
        """Test cropping 16:9 to 9:16 (center)."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1920,
            source_height=1080,
            target_ratio=AspectRatio.PORTRAIT_9_16,
            x_offset=0.5,
            y_offset=0.5,
        )

        # For 9:16 from 1920x1080:
        # crop_height = 1080 (already even)
        # crop_width = 1080 * 9/16 = 607.5 -> 606 (even)
        assert crop_h == 1080
        assert crop_w == 606
        # Center x position: (1920 - 606) * 0.5 = 657
        assert x == 657
        assert y == 0
        # Both dimensions must be even
        assert crop_w % 2 == 0
        assert crop_h % 2 == 0

    def test_landscape_to_portrait_left_crop(self):
        """Test cropping with left offset."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1920,
            source_height=1080,
            target_ratio=AspectRatio.PORTRAIT_9_16,
            x_offset=0.0,
            y_offset=0.5,
        )

        assert crop_h == 1080
        assert x == 0  # Left edge

    def test_landscape_to_portrait_right_crop(self):
        """Test cropping with right offset."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1920,
            source_height=1080,
            target_ratio=AspectRatio.PORTRAIT_9_16,
            x_offset=1.0,
            y_offset=0.5,
        )

        # x should be at right edge
        assert crop_h == 1080
        assert x == 1920 - crop_w

    def test_landscape_to_square(self):
        """Test cropping 16:9 to 1:1."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1920,
            source_height=1080,
            target_ratio=AspectRatio.SQUARE_1_1,
            x_offset=0.5,
            y_offset=0.5,
        )

        # Square from 1920x1080: use full height
        assert crop_h == 1080
        assert crop_w == 1080
        # Center: (1920 - 1080) / 2 = 420
        assert x == 420
        assert y == 0

    def test_portrait_to_portrait(self):
        """Test cropping portrait source to different portrait ratio."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1080,
            source_height=1920,
            target_ratio=AspectRatio.PORTRAIT_4_5,
            x_offset=0.5,
            y_offset=0.5,
        )

        # 4:5 is wider than 9:16, so crop vertically
        # crop_width = 1080
        # crop_height = 1080 / (4/5) = 1350
        assert crop_w == 1080
        assert crop_h == 1350
        assert x == 0
        # Center y: (1920 - 1350) * 0.5 = 285
        assert y == 285

    def test_even_dimensions(self):
        """Test that crop dimensions are always even."""
        # Use odd source dimensions
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1921,
            source_height=1081,
            target_ratio=AspectRatio.PORTRAIT_9_16,
        )

        # Both dimensions should be even for video encoding
        assert crop_w % 2 == 0
        assert crop_h % 2 == 0
        # Height is source-1 (1080), width calculated from that
        assert crop_h == 1080
        assert crop_w == 606

    def test_4_3_to_9_16(self):
        """Test cropping 4:3 to 9:16."""
        crop_w, crop_h, x, y = calculate_crop_region(
            source_width=1440,
            source_height=1080,
            target_ratio=AspectRatio.PORTRAIT_9_16,
            x_offset=0.5,
            y_offset=0.5,
        )

        # 4:3 is wider than 9:16, crop horizontally
        assert crop_h == 1080
        # crop_width = 1080 * 9/16 = 607.5 -> 606
        assert crop_w == 606


class TestPortraitConverter:
    """Tests for PortraitConverter class."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "/usr/bin/ffmpeg"
            converter = PortraitConverter()

            assert converter.ffmpeg_path == "/usr/bin/ffmpeg"

    def test_init_custom_path(self):
        """Test initialization with custom FFmpeg path."""
        converter = PortraitConverter(ffmpeg_path="/custom/ffmpeg")
        assert converter.ffmpeg_path == "/custom/ffmpeg"

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_basic(self, mock_run, tmp_path):
        """Test basic video conversion."""
        mock_run.return_value = Mock(returncode=0, stderr="")

        # Create mock input file
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.mp4"

        # Mock video info
        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            result = converter.convert(input_file, output_file)

            assert result == output_file
            mock_run.assert_called_once()

            # Check FFmpeg command
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd[0]
            assert "-i" in cmd
            assert "-vf" in cmd
            assert "-c:v" in cmd
            assert "libx264" in cmd

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_with_config(self, mock_run, tmp_path):
        """Test conversion with custom config."""
        mock_run.return_value = Mock(returncode=0, stderr="")

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.mp4"

        config = PortraitConfig(
            target_ratio=AspectRatio.PORTRAIT_4_5,
            video_crf=18,
            video_preset="slow",
        )

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            converter.convert(input_file, output_file, config=config)

            cmd = mock_run.call_args[0][0]
            assert "-crf" in cmd
            crf_idx = cmd.index("-crf")
            assert cmd[crf_idx + 1] == "18"
            assert "-preset" in cmd
            preset_idx = cmd.index("-preset")
            assert cmd[preset_idx + 1] == "slow"

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_creates_output_dir(self, mock_run, tmp_path):
        """Test that output directory is created."""
        mock_run.return_value = Mock(returncode=0, stderr="")

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "new" / "nested" / "dir"
        output_file = output_dir / "output.mp4"

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            converter.convert(input_file, output_file)

            assert output_dir.exists()

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_ffmpeg_error(self, mock_run, tmp_path):
        """Test handling of FFmpeg errors."""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Invalid input file",
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.mp4"

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            with pytest.raises(RuntimeError, match="FFmpeg error"):
                converter.convert(input_file, output_file)

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_faststart(self, mock_run, tmp_path):
        """Test faststart option."""
        mock_run.return_value = Mock(returncode=0, stderr="")

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.mp4"

        config = PortraitConfig(faststart=True)

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            converter.convert(input_file, output_file, config=config)

            cmd = mock_run.call_args[0][0]
            assert "-movflags" in cmd
            movflags_idx = cmd.index("-movflags")
            assert "+faststart" in cmd[movflags_idx + 1]

    def test_get_optimal_crop(self, tmp_path):
        """Test get_optimal_crop method."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            crop_w, crop_h, x, y = converter.get_optimal_crop(input_file)

            # Default is 9:16
            assert crop_h == 1080
            assert crop_w == 606


class TestPlatformConfigs:
    """Tests for platform-specific configurations."""

    def test_youtube_shorts_config(self):
        """Test YouTube Shorts configuration."""
        config = YOUTUBE_SHORTS_CONFIG

        assert config.target_ratio == AspectRatio.PORTRAIT_9_16
        assert config.target_width == 1080
        assert config.target_height == 1920
        assert config.video_crf == 20
        assert config.video_preset == "slow"

    def test_tiktok_config(self):
        """Test TikTok configuration."""
        config = TIKTOK_CONFIG

        assert config.target_ratio == AspectRatio.PORTRAIT_9_16
        assert config.target_width == 1080
        assert config.video_crf == 18  # Higher quality

    def test_instagram_reels_config(self):
        """Test Instagram Reels configuration."""
        config = INSTAGRAM_REELS_CONFIG

        assert config.target_ratio == AspectRatio.PORTRAIT_9_16
        assert config.target_width == 1080
        assert config.faststart is True

    def test_instagram_feed_config(self):
        """Test Instagram feed configuration."""
        config = INSTAGRAM_FEED_CONFIG

        assert config.target_ratio == AspectRatio.PORTRAIT_4_5
        assert config.target_width == 1080
        assert config.target_height == 1350

    def test_linkedin_config(self):
        """Test LinkedIn configuration."""
        config = LINKEDIN_CONFIG

        assert config.target_ratio == AspectRatio.SQUARE_1_1
        assert config.target_width == 1080
        assert config.target_height == 1080


class TestConvertWithCaptions:
    """Tests for convert_with_captions method."""

    @patch("clip_video.video.portrait.subprocess.run")
    def test_convert_with_captions(self, mock_run, tmp_path):
        """Test conversion with burned-in captions."""
        mock_run.return_value = Mock(returncode=0, stderr="")

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_file = tmp_path / "output.mp4"

        # Create mock caption track
        mock_caption = Mock()
        mock_caption.text = "Hello world"
        mock_caption.start_time = 0.0
        mock_caption.end_time = 2.0
        mock_caption.style = None

        mock_track = Mock()
        mock_track.captions = [mock_caption]

        with patch("clip_video.video.portrait.get_ffmpeg_path") as mock_get:
            mock_get.return_value = "ffmpeg"
            converter = PortraitConverter()
            converter.ffmpeg_wrapper = Mock()
            converter.ffmpeg_wrapper.get_video_info.return_value = Mock(
                width=1920,
                height=1080,
            )

            result = converter.convert_with_captions(
                input_file,
                output_file,
                caption_track=mock_track,
            )

            assert result == output_file
            mock_run.assert_called_once()

            # Check that drawtext filter is in command
            cmd = mock_run.call_args[0][0]
            vf_idx = cmd.index("-vf")
            filter_str = cmd[vf_idx + 1]
            assert "drawtext" in filter_str
            assert "crop=" in filter_str
            assert "scale=" in filter_str
