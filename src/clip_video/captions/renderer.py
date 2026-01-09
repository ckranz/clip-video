"""Caption rendering using FFmpeg.

Provides functionality to burn captions into videos using FFmpeg's
drawtext filter or ASS subtitle rendering.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Iterator

from clip_video.captions.styles import CaptionStyle, DEFAULT_STYLE
from clip_video.ffmpeg_binary import get_ffmpeg_path
from clip_video.ffmpeg import FFmpegWrapper, VideoInfo


@dataclass
class Caption:
    """A single caption to be rendered.

    Attributes:
        text: Caption text content
        start_time: Start time in seconds
        end_time: End time in seconds
        style: Optional per-caption style override
    """

    text: str
    start_time: float
    end_time: float
    style: CaptionStyle | None = None

    @property
    def duration(self) -> float:
        """Duration of the caption in seconds."""
        return self.end_time - self.start_time

    def get_wrapped_text(self, max_chars_per_line: int = 40) -> str:
        """Wrap text to fit within max characters per line.

        Args:
            max_chars_per_line: Maximum characters per line

        Returns:
            Wrapped text with newlines
        """
        words = self.text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)


@dataclass
class CaptionTrack:
    """A collection of captions for a video.

    Attributes:
        captions: List of captions in time order
        default_style: Default style for all captions
        video_duration: Total video duration in seconds
    """

    captions: List[Caption] = field(default_factory=list)
    default_style: CaptionStyle = field(default_factory=lambda: DEFAULT_STYLE)
    video_duration: float = 0.0

    def add_caption(
        self,
        text: str,
        start_time: float,
        end_time: float,
        style: CaptionStyle | None = None,
    ) -> Caption:
        """Add a caption to the track.

        Args:
            text: Caption text
            start_time: Start time in seconds
            end_time: End time in seconds
            style: Optional style override

        Returns:
            The created Caption
        """
        caption = Caption(
            text=text,
            start_time=start_time,
            end_time=end_time,
            style=style,
        )
        self.captions.append(caption)
        return caption

    def sort_by_time(self) -> None:
        """Sort captions by start time."""
        self.captions.sort(key=lambda c: c.start_time)

    @classmethod
    def from_transcript_segments(
        cls,
        segments: list[dict],
        style: CaptionStyle | None = None,
    ) -> "CaptionTrack":
        """Create a caption track from transcript segments.

        Args:
            segments: List of segments with 'text', 'start', 'end' keys
            style: Optional default style

        Returns:
            CaptionTrack with captions for each segment
        """
        track = cls(default_style=style or DEFAULT_STYLE)

        for segment in segments:
            track.add_caption(
                text=segment["text"],
                start_time=segment["start"],
                end_time=segment["end"],
            )

        track.sort_by_time()
        return track

    def to_srt(self) -> str:
        """Export captions as SRT format.

        Returns:
            SRT formatted string
        """
        lines = []
        for i, caption in enumerate(self.captions, 1):
            start = self._format_srt_time(caption.start_time)
            end = self._format_srt_time(caption.end_time)
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(caption.text)
            lines.append("")

        return "\n".join(lines)

    def to_ass(self) -> str:
        """Export captions as ASS/SSA format.

        Returns:
            ASS formatted string
        """
        style = self.default_style

        # Build ASS header
        header = f"""[Script Info]
Title: Generated Captions
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_family},{style.font_size},&H00{style.font_color},&H00FFFFFF,&H00{style.border_color},&H80{style.background_color},{1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.border_width},2,2,{style.margin_x},{style.margin_x},{style.margin_y},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        for caption in self.captions:
            start = self._format_ass_time(caption.start_time)
            end = self._format_ass_time(caption.end_time)
            text = caption.text.replace("\n", "\\N")
            if style.uppercase:
                text = text.upper()
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        return header + "\n".join(events)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """Format time for ASS (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


class CaptionRenderer:
    """Renders captions onto videos using FFmpeg.

    Supports two rendering methods:
    1. drawtext filter: Direct text rendering (simple, cross-platform)
    2. ASS subtitles: Advanced styling with word wrapping

    Example usage:
        renderer = CaptionRenderer()
        track = CaptionTrack()
        track.add_caption("Hello World!", 0.0, 3.0)

        renderer.render(
            input_path=Path("video.mp4"),
            output_path=Path("video_captioned.mp4"),
            caption_track=track,
        )
    """

    def __init__(self, ffmpeg_path: str | None = None):
        """Initialize the renderer.

        Args:
            ffmpeg_path: Path to FFmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
        self.ffmpeg_wrapper = FFmpegWrapper()

    def render(
        self,
        input_path: Path | str,
        output_path: Path | str,
        caption_track: CaptionTrack,
        method: str = "drawtext",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
    ) -> Path:
        """Render captions onto a video.

        Args:
            input_path: Input video path
            output_path: Output video path
            caption_track: Caption track to render
            method: Rendering method ("drawtext" or "ass")
            video_codec: Video codec for output
            audio_codec: Audio codec for output
            crf: Quality level for H.264

        Returns:
            Path to output video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Get video info for sizing calculations
        video_info = self.ffmpeg_wrapper.get_video_info(input_path)

        if method == "ass":
            return self._render_with_ass(
                input_path, output_path, caption_track, video_info,
                video_codec, audio_codec, crf
            )
        else:
            return self._render_with_drawtext(
                input_path, output_path, caption_track, video_info,
                video_codec, audio_codec, crf
            )

    def _render_with_drawtext(
        self,
        input_path: Path,
        output_path: Path,
        caption_track: CaptionTrack,
        video_info: VideoInfo,
        video_codec: str,
        audio_codec: str,
        crf: int,
    ) -> Path:
        """Render using FFmpeg drawtext filter."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build filter complex with all captions
        filter_parts = []
        style = caption_track.default_style

        for caption in caption_track.captions:
            cap_style = caption.style or style
            params = cap_style.to_drawtext_params(video_info.width, video_info.height)

            # Escape text for FFmpeg
            text = caption.text
            if cap_style.uppercase:
                text = text.upper()
            text = self._escape_drawtext(text)

            # Build drawtext filter
            param_str = ":".join(f"{k}={v}" for k, v in params.items())
            enable = f"between(t,{caption.start_time},{caption.end_time})"

            filter_parts.append(
                f"drawtext=text='{text}':{param_str}:enable='{enable}'"
            )

        if not filter_parts:
            # No captions, just copy
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-y",
                str(output_path),
            ]
        else:
            filter_complex = ",".join(filter_parts)
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-vf", filter_complex,
                "-c:v", video_codec,
                "-crf", str(crf),
                "-c:a", audio_codec,
                "-y",
                str(output_path),
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        return output_path

    def _render_with_ass(
        self,
        input_path: Path,
        output_path: Path,
        caption_track: CaptionTrack,
        video_info: VideoInfo,
        video_codec: str,
        audio_codec: str,
        crf: int,
    ) -> Path:
        """Render using ASS subtitle overlay."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary ASS file
        ass_content = caption_track.to_ass()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ass",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(ass_content)
            ass_path = f.name

        try:
            # FFmpeg command with ASS filter
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-vf", f"ass={ass_path}",
                "-c:v", video_codec,
                "-crf", str(crf),
                "-c:a", audio_codec,
                "-y",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")

        finally:
            # Clean up temp file
            Path(ass_path).unlink(missing_ok=True)

        return output_path

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """Escape special characters for FFmpeg drawtext filter.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        # Escape single quotes, colons, and backslashes
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "'\\''")
        text = text.replace(":", "\\:")
        text = text.replace("%", "\\%")
        return text

    def create_srt_file(
        self,
        caption_track: CaptionTrack,
        output_path: Path,
    ) -> Path:
        """Export caption track as SRT file.

        Args:
            caption_track: Caption track to export
            output_path: Output file path

        Returns:
            Path to created SRT file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        srt_content = caption_track.to_srt()
        output_path.write_text(srt_content, encoding="utf-8")
        return output_path

    def create_ass_file(
        self,
        caption_track: CaptionTrack,
        output_path: Path,
    ) -> Path:
        """Export caption track as ASS file.

        Args:
            caption_track: Caption track to export
            output_path: Output file path

        Returns:
            Path to created ASS file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_content = caption_track.to_ass()
        output_path.write_text(ass_content, encoding="utf-8")
        return output_path
