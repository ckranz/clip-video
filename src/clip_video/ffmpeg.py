"""FFmpeg wrapper for video clip extraction.

Provides high-level interface for extracting clips from videos with
configurable start/end times, padding, and portrait crop functionality.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from clip_video.ffmpeg_binary import get_ffmpeg_path, get_ffprobe_path, FFmpegConfig
from clip_video.models.brand import CropRegion

if TYPE_CHECKING:
    pass


class FFmpegError(Exception):
    """Base exception for FFmpeg-related errors."""

    pass


class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg executable is not found."""

    pass


class InvalidVideoError(FFmpegError):
    """Raised when the input video file is invalid or unreadable."""

    pass


class ExtractionError(FFmpegError):
    """Raised when clip extraction fails."""

    pass


class EncodingMode(str, Enum):
    """Video encoding mode for clip extraction."""

    COPY = "copy"  # Stream copy - fast, no re-encoding
    REENCODE = "reencode"  # Re-encode - slower, enables filtering/format conversion


@dataclass
class VideoInfo:
    """Information about a video file."""

    duration: float  # Total duration in seconds
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    has_audio: bool


class ClipPadding(BaseModel):
    """Padding configuration for clip extraction.

    Adds extra time before and after the specified clip boundaries
    to ensure smooth transitions and capture context.
    """

    before: float = Field(default=0.3, ge=0.0, description="Seconds to add before clip start")
    after: float = Field(default=0.5, ge=0.0, description="Seconds to add after clip end")


class ExtractionConfig(BaseModel):
    """Configuration for clip extraction."""

    padding: ClipPadding = Field(default_factory=ClipPadding)
    mode: EncodingMode = EncodingMode.COPY
    crop_region: CropRegion | None = None
    output_format: str = "mp4"

    # Video encoding settings (used when mode=REENCODE)
    video_codec: str = "libx264"
    video_bitrate: str | None = None  # e.g., "5M" for 5 Mbps
    video_crf: int = 23  # Quality for H.264 (lower = better, 18-28 typical)
    video_preset: str = "medium"  # Speed/quality tradeoff

    # Audio encoding settings (used when mode=REENCODE)
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    # Additional FFmpeg options
    extra_input_args: list[str] = Field(default_factory=list)
    extra_output_args: list[str] = Field(default_factory=list)


class FFmpegWrapper:
    """Wrapper for FFmpeg video clip extraction.

    Provides high-level methods for extracting clips from videos with
    configurable timing, padding, and cropping options. Outputs are
    formatted for DaVinci Resolve compatibility (MP4/H.264/AAC).
    """

    def __init__(self, config: FFmpegConfig | None = None) -> None:
        """Initialize FFmpeg wrapper.

        Args:
            config: Optional FFmpeg binary configuration.

        Raises:
            FFmpegNotFoundError: If FFmpeg is not available.
        """
        self._config = config or FFmpegConfig()
        self._ffmpeg_path = get_ffmpeg_path(self._config)
        self._ffprobe_path = get_ffprobe_path(self._config)

        if self._ffmpeg_path is None:
            raise FFmpegNotFoundError(
                "FFmpeg not found. Please install imageio-ffmpeg or add FFmpeg to PATH."
            )

    @property
    def ffmpeg_path(self) -> str:
        """Get path to FFmpeg executable."""
        return self._ffmpeg_path

    @property
    def ffprobe_path(self) -> str | None:
        """Get path to FFprobe executable."""
        return self._ffprobe_path

    def _get_subprocess_flags(self) -> int:
        """Get platform-specific subprocess creation flags."""
        if platform.system() == "Windows":
            return subprocess.CREATE_NO_WINDOW
        return 0

    def _run_ffmpeg(
        self,
        args: list[str],
        timeout: int = 300,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run FFmpeg with the given arguments.

        Args:
            args: Command-line arguments (excluding ffmpeg executable).
            timeout: Timeout in seconds.
            check: Whether to raise on non-zero exit code.

        Returns:
            CompletedProcess result.

        Raises:
            ExtractionError: If FFmpeg fails and check=True.
        """
        cmd = [self._ffmpeg_path] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=self._get_subprocess_flags(),
            )

            if check and result.returncode != 0:
                # FFmpeg outputs to stderr even for version info
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise ExtractionError(f"FFmpeg failed: {error_msg}")

            return result

        except subprocess.TimeoutExpired as e:
            raise ExtractionError(f"FFmpeg timed out after {timeout} seconds") from e
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(f"FFmpeg not found at {self._ffmpeg_path}") from e
        except OSError as e:
            raise ExtractionError(f"Failed to run FFmpeg: {e}") from e

    def _run_ffprobe(
        self,
        args: list[str],
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        """Run FFprobe with the given arguments.

        Args:
            args: Command-line arguments (excluding ffprobe executable).
            timeout: Timeout in seconds.

        Returns:
            CompletedProcess result.

        Raises:
            FFmpegNotFoundError: If FFprobe is not available.
            ExtractionError: If FFprobe fails.
        """
        if self._ffprobe_path is None:
            raise FFmpegNotFoundError(
                "FFprobe not found. Please install FFprobe to enable video inspection."
            )

        cmd = [self._ffprobe_path] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=self._get_subprocess_flags(),
            )
            return result

        except subprocess.TimeoutExpired as e:
            raise ExtractionError(f"FFprobe timed out after {timeout} seconds") from e
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(f"FFprobe not found at {self._ffprobe_path}") from e
        except OSError as e:
            raise ExtractionError(f"Failed to run FFprobe: {e}") from e

    def get_video_info(self, video_path: str | Path) -> VideoInfo:
        """Get information about a video file.

        Args:
            video_path: Path to video file.

        Returns:
            VideoInfo with duration, dimensions, fps, and codecs.

        Raises:
            InvalidVideoError: If the file is not a valid video.
            FFmpegNotFoundError: If FFprobe is not available.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise InvalidVideoError(f"Video file not found: {video_path}")

        args = [
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = self._run_ffprobe(args)

        if result.returncode != 0:
            raise InvalidVideoError(
                f"Failed to read video file: {video_path}\n{result.stderr}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise InvalidVideoError(f"Failed to parse video info: {e}") from e

        # Find video and audio streams
        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if video_stream is None:
            raise InvalidVideoError(f"No video stream found in: {video_path}")

        # Extract video info
        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Calculate FPS from frame rate fraction
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

        video_codec = video_stream.get("codec_name", "unknown")
        audio_codec = audio_stream.get("codec_name") if audio_stream else None

        return VideoInfo(
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video_codec,
            audio_codec=audio_codec,
            has_audio=audio_stream is not None,
        )

    def validate_video(self, video_path: str | Path) -> tuple[bool, str]:
        """Validate that a file is a readable video.

        Args:
            video_path: Path to video file.

        Returns:
            Tuple of (is_valid, message).
        """
        try:
            info = self.get_video_info(video_path)
            if info.duration <= 0:
                return False, "Video has zero or negative duration"
            if info.width <= 0 or info.height <= 0:
                return False, "Video has invalid dimensions"
            return True, f"Valid video: {info.width}x{info.height}, {info.duration:.2f}s"
        except InvalidVideoError as e:
            return False, str(e)
        except FFmpegNotFoundError as e:
            return False, str(e)

    def _calculate_adjusted_times(
        self,
        start_time: float,
        end_time: float,
        padding: ClipPadding,
        video_duration: float,
    ) -> tuple[float, float]:
        """Calculate adjusted start/end times with padding.

        Args:
            start_time: Original start time in seconds.
            end_time: Original end time in seconds.
            padding: Padding configuration.
            video_duration: Total video duration.

        Returns:
            Tuple of (adjusted_start, adjusted_end).
        """
        adjusted_start = max(0.0, start_time - padding.before)
        adjusted_end = min(video_duration, end_time + padding.after)
        return adjusted_start, adjusted_end

    def _build_filter_complex(
        self,
        crop_region: CropRegion | None,
    ) -> str | None:
        """Build FFmpeg filter_complex string.

        Args:
            crop_region: Optional crop region for portrait conversion.

        Returns:
            Filter string or None if no filters needed.
        """
        filters = []

        if crop_region is not None:
            filters.append(crop_region.to_ffmpeg_filter())

        if not filters:
            return None

        return ",".join(filters)

    def _build_extraction_args(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        duration: float,
        config: ExtractionConfig,
    ) -> list[str]:
        """Build FFmpeg arguments for clip extraction.

        Args:
            input_path: Path to source video.
            output_path: Path for output file.
            start_time: Start time in seconds.
            duration: Duration in seconds.
            config: Extraction configuration.

        Returns:
            List of FFmpeg arguments.
        """
        args = []

        # Overwrite output without asking
        args.extend(["-y"])

        # Add any extra input args
        args.extend(config.extra_input_args)

        # Input seeking (before -i for fast seeking)
        args.extend(["-ss", f"{start_time:.3f}"])

        # Input file
        args.extend(["-i", str(input_path)])

        # Duration
        args.extend(["-t", f"{duration:.3f}"])

        # Build filter if needed (crop requires re-encoding)
        filter_str = self._build_filter_complex(config.crop_region)

        # Determine if we need to re-encode
        needs_reencode = (
            config.mode == EncodingMode.REENCODE or
            filter_str is not None
        )

        if needs_reencode:
            # Video encoding
            args.extend(["-c:v", config.video_codec])

            if config.video_bitrate:
                args.extend(["-b:v", config.video_bitrate])
            else:
                args.extend(["-crf", str(config.video_crf)])

            args.extend(["-preset", config.video_preset])

            # Pixel format for compatibility
            args.extend(["-pix_fmt", "yuv420p"])

            # Audio encoding
            args.extend(["-c:a", config.audio_codec])
            args.extend(["-b:a", config.audio_bitrate])

            # Apply filters
            if filter_str:
                args.extend(["-vf", filter_str])

        else:
            # Copy mode - fast, no re-encoding
            args.extend(["-c:v", "copy"])
            args.extend(["-c:a", "copy"])

        # Output container options for DaVinci Resolve compatibility
        args.extend(["-movflags", "+faststart"])

        # Add any extra output args
        args.extend(config.extra_output_args)

        # Output file
        args.append(str(output_path))

        return args

    def extract_clip(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float,
        end_time: float,
        config: ExtractionConfig | None = None,
    ) -> Path:
        """Extract a clip from a video file.

        Args:
            input_path: Path to source video.
            output_path: Path for output file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            config: Extraction configuration.

        Returns:
            Path to the extracted clip.

        Raises:
            InvalidVideoError: If input video is invalid.
            ExtractionError: If extraction fails.
            ValueError: If time range is invalid.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if config is None:
            config = ExtractionConfig()

        # Validate input video
        is_valid, msg = self.validate_video(input_path)
        if not is_valid:
            raise InvalidVideoError(msg)

        # Get video info for duration
        video_info = self.get_video_info(input_path)

        # Validate time range
        if start_time < 0:
            raise ValueError(f"Start time cannot be negative: {start_time}")
        if end_time <= start_time:
            raise ValueError(f"End time must be after start time: {start_time} -> {end_time}")
        if start_time >= video_info.duration:
            raise ValueError(f"Start time {start_time} exceeds video duration {video_info.duration}")

        # Calculate adjusted times with padding
        adjusted_start, adjusted_end = self._calculate_adjusted_times(
            start_time,
            end_time,
            config.padding,
            video_info.duration,
        )

        duration = adjusted_end - adjusted_start

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build and run FFmpeg command
        args = self._build_extraction_args(
            input_path,
            output_path,
            adjusted_start,
            duration,
            config,
        )

        self._run_ffmpeg(args)

        # Verify output was created
        if not output_path.exists():
            raise ExtractionError(f"Output file was not created: {output_path}")

        return output_path

    def extract_clip_portrait(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float,
        end_time: float,
        crop_region: CropRegion | None = None,
        padding: ClipPadding | None = None,
    ) -> Path:
        """Extract a clip and convert to portrait orientation.

        Convenience method that sets up extraction with portrait crop.

        Args:
            input_path: Path to source video.
            output_path: Path for output file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            crop_region: Custom crop region (uses default if None).
            padding: Custom padding (uses default if None).

        Returns:
            Path to the extracted portrait clip.
        """
        config = ExtractionConfig(
            mode=EncodingMode.REENCODE,  # Required for cropping
            crop_region=crop_region or CropRegion(),
            padding=padding or ClipPadding(),
        )

        return self.extract_clip(input_path, output_path, start_time, end_time, config)

    def extract_clip_fast(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start_time: float,
        end_time: float,
        padding: ClipPadding | None = None,
    ) -> Path:
        """Extract a clip using stream copy (fast, no re-encoding).

        Convenience method for quick extraction without any filters.
        May not be frame-accurate due to keyframe alignment.

        Args:
            input_path: Path to source video.
            output_path: Path for output file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            padding: Custom padding (uses default if None).

        Returns:
            Path to the extracted clip.
        """
        config = ExtractionConfig(
            mode=EncodingMode.COPY,
            padding=padding or ClipPadding(),
        )

        return self.extract_clip(input_path, output_path, start_time, end_time, config)

    def get_thumbnail(
        self,
        video_path: str | Path,
        output_path: str | Path,
        timestamp: float = 0.0,
        width: int = 320,
    ) -> Path:
        """Extract a thumbnail image from a video.

        Args:
            video_path: Path to video file.
            output_path: Path for output image (should be .jpg or .png).
            timestamp: Time in seconds to capture thumbnail.
            width: Output width (height auto-calculated to maintain aspect).

        Returns:
            Path to the thumbnail image.

        Raises:
            InvalidVideoError: If input video is invalid.
            ExtractionError: If thumbnail extraction fails.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # Validate input
        is_valid, msg = self.validate_video(video_path)
        if not is_valid:
            raise InvalidVideoError(msg)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            str(output_path),
        ]

        self._run_ffmpeg(args, timeout=30)

        if not output_path.exists():
            raise ExtractionError(f"Thumbnail was not created: {output_path}")

        return output_path


def create_ffmpeg_wrapper(config: FFmpegConfig | None = None) -> FFmpegWrapper:
    """Factory function to create an FFmpegWrapper instance.

    Args:
        config: Optional FFmpeg binary configuration.

    Returns:
        FFmpegWrapper instance.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not available.
    """
    return FFmpegWrapper(config)
