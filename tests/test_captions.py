"""Tests for caption rendering."""

import pytest
from pathlib import Path

from clip_video.captions.styles import (
    CaptionStyle,
    CaptionPosition,
    DEFAULT_STYLE,
    YOUTUBE_SHORTS_STYLE,
    TIKTOK_STYLE,
)
from clip_video.captions.renderer import (
    Caption,
    CaptionTrack,
    CaptionRenderer,
)


class TestCaptionStyle:
    """Tests for CaptionStyle class."""

    def test_default_values(self):
        """Test default style values."""
        style = CaptionStyle()

        assert style.font_family == "Arial"
        assert style.font_size == 48
        assert style.font_color == "FFFFFF"
        assert style.position == CaptionPosition.LOWER_THIRD

    def test_get_y_position_top(self):
        """Test Y position calculation for top."""
        style = CaptionStyle(position=CaptionPosition.TOP, margin_y=50)
        y = style.get_y_position(1080)

        assert y == 50

    def test_get_y_position_bottom(self):
        """Test Y position calculation for bottom."""
        style = CaptionStyle(
            position=CaptionPosition.BOTTOM,
            margin_y=50,
            font_size=40,
        )
        y = style.get_y_position(1080)

        assert y == 1080 - 50 - 40

    def test_get_y_position_center(self):
        """Test Y position calculation for center."""
        style = CaptionStyle(position=CaptionPosition.CENTER)
        y = style.get_y_position(1080)

        assert y == 540

    def test_get_y_position_lower_third(self):
        """Test Y position calculation for lower third."""
        style = CaptionStyle(position=CaptionPosition.LOWER_THIRD)
        y = style.get_y_position(1080)

        assert y == int(1080 * 0.75)

    def test_get_ffmpeg_color(self):
        """Test FFmpeg color formatting."""
        style = CaptionStyle()
        color = style.get_ffmpeg_color("FFFFFF", 1.0)

        assert color == "0xFFFFFFFF"

    def test_get_ffmpeg_color_with_opacity(self):
        """Test FFmpeg color with opacity."""
        style = CaptionStyle()
        color = style.get_ffmpeg_color("000000", 0.5)

        # 0.5 opacity = 127 in hex = 7F
        assert "0x000000" in color

    def test_to_drawtext_params(self):
        """Test converting style to drawtext parameters."""
        style = CaptionStyle(font_size=60, border_width=3)
        params = style.to_drawtext_params(1920, 1080)

        assert params["fontsize"] == 60
        assert params["borderw"] == 3
        assert "fontcolor" in params

    def test_pre_built_styles(self):
        """Test pre-built style configurations."""
        assert YOUTUBE_SHORTS_STYLE.font_size == 72
        assert TIKTOK_STYLE.uppercase is True
        assert TIKTOK_STYLE.position == CaptionPosition.CENTER


class TestCaption:
    """Tests for Caption class."""

    def test_duration(self):
        """Test duration calculation."""
        caption = Caption(
            text="Hello World",
            start_time=1.0,
            end_time=3.5,
        )

        assert caption.duration == 2.5

    def test_get_wrapped_text_short(self):
        """Test text that doesn't need wrapping."""
        caption = Caption(
            text="Short text",
            start_time=0,
            end_time=1,
        )

        wrapped = caption.get_wrapped_text(max_chars_per_line=40)
        assert wrapped == "Short text"
        assert "\n" not in wrapped

    def test_get_wrapped_text_long(self):
        """Test text that needs wrapping."""
        caption = Caption(
            text="This is a much longer piece of text that needs to be wrapped",
            start_time=0,
            end_time=1,
        )

        wrapped = caption.get_wrapped_text(max_chars_per_line=20)

        assert "\n" in wrapped
        for line in wrapped.split("\n"):
            assert len(line) <= 25  # Allow some flexibility

    def test_style_override(self):
        """Test per-caption style override."""
        custom_style = CaptionStyle(font_size=72)
        caption = Caption(
            text="Styled text",
            start_time=0,
            end_time=1,
            style=custom_style,
        )

        assert caption.style.font_size == 72


class TestCaptionTrack:
    """Tests for CaptionTrack class."""

    def test_add_caption(self):
        """Test adding captions."""
        track = CaptionTrack()
        track.add_caption("First caption", 0.0, 2.0)
        track.add_caption("Second caption", 2.5, 4.0)

        assert len(track.captions) == 2

    def test_sort_by_time(self):
        """Test sorting captions by time."""
        track = CaptionTrack()
        track.add_caption("Second", 2.0, 3.0)
        track.add_caption("First", 0.0, 1.0)
        track.add_caption("Third", 4.0, 5.0)

        track.sort_by_time()

        assert track.captions[0].text == "First"
        assert track.captions[1].text == "Second"
        assert track.captions[2].text == "Third"

    def test_from_transcript_segments(self):
        """Test creating from transcript segments."""
        segments = [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "World", "start": 1.5, "end": 2.5},
        ]

        track = CaptionTrack.from_transcript_segments(segments)

        assert len(track.captions) == 2
        assert track.captions[0].text == "Hello"
        assert track.captions[1].start_time == 1.5

    def test_to_srt(self):
        """Test SRT export."""
        track = CaptionTrack()
        track.add_caption("First caption", 0.0, 2.0)
        track.add_caption("Second caption", 2.5, 4.5)

        srt = track.to_srt()

        assert "1" in srt
        assert "2" in srt
        assert "First caption" in srt
        assert "Second caption" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt

    def test_to_ass(self):
        """Test ASS export."""
        track = CaptionTrack()
        track.add_caption("Test caption", 0.0, 2.0)

        ass = track.to_ass()

        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Test caption" in ass

    def test_format_srt_time(self):
        """Test SRT time formatting."""
        time_str = CaptionTrack._format_srt_time(3661.5)  # 1:01:01.500
        assert time_str == "01:01:01,500"

    def test_format_ass_time(self):
        """Test ASS time formatting."""
        time_str = CaptionTrack._format_ass_time(3661.5)  # 1:01:01.50
        assert time_str == "1:01:01.50"


class TestCaptionRenderer:
    """Tests for CaptionRenderer class."""

    def test_escape_drawtext(self):
        """Test escaping special characters."""
        escaped = CaptionRenderer._escape_drawtext("Hello: World's 50%")

        assert "\\:" in escaped
        assert "\\%" in escaped

    def test_create_srt_file(self, tmp_path):
        """Test creating SRT file."""
        renderer = CaptionRenderer()
        track = CaptionTrack()
        track.add_caption("Test caption", 0.0, 2.0)

        output_path = tmp_path / "captions.srt"
        result = renderer.create_srt_file(track, output_path)

        assert result.exists()
        content = result.read_text()
        assert "Test caption" in content
        assert "-->" in content

    def test_create_ass_file(self, tmp_path):
        """Test creating ASS file."""
        renderer = CaptionRenderer()
        track = CaptionTrack()
        track.add_caption("Test caption", 0.0, 2.0)

        output_path = tmp_path / "captions.ass"
        result = renderer.create_ass_file(track, output_path)

        assert result.exists()
        content = result.read_text()
        assert "[Script Info]" in content
        assert "Test caption" in content


class TestCaptionPositionEnum:
    """Tests for CaptionPosition enum."""

    def test_values(self):
        """Test enum values."""
        assert CaptionPosition.TOP.value == "top"
        assert CaptionPosition.CENTER.value == "center"
        assert CaptionPosition.BOTTOM.value == "bottom"
        assert CaptionPosition.LOWER_THIRD.value == "lower_third"
