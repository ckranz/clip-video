"""Portrait video conversion for social media.

Provides functionality to convert landscape videos to portrait (9:16)
format optimized for YouTube Shorts, TikTok, and Instagram Reels.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

from clip_video.ffmpeg_binary import get_ffmpeg_path
from clip_video.ffmpeg import FFmpegWrapper, VideoInfo

if TYPE_CHECKING:
    from clip_video.config import LogoSettings


class AspectRatio(str, Enum):
    """Common aspect ratios for video output."""

    PORTRAIT_9_16 = "9:16"  # YouTube Shorts, TikTok, Reels
    PORTRAIT_4_5 = "4:5"  # Instagram feed
    SQUARE_1_1 = "1:1"  # Instagram square
    LANDSCAPE_16_9 = "16:9"  # Standard widescreen
    LANDSCAPE_4_3 = "4:3"  # Classic TV

    @property
    def width_ratio(self) -> float:
        """Get width portion of ratio."""
        w, h = self.value.split(":")
        return float(w) / float(h)

    @property
    def is_portrait(self) -> bool:
        """Check if this is a portrait orientation."""
        return self.width_ratio < 1.0


@dataclass
class LogoOverlayConfig:
    """Configuration for logo overlay on video.

    Attributes:
        enabled: Whether to add logo overlay
        logo_path: Path to the logo image file
        position: Position on video (top-left, top-center, top-right,
                  bottom-left, bottom-center, bottom-right)
        height_percent: Logo height as percentage of video height (0.0-1.0)
        opacity: Logo opacity (0.0-1.0, 1.0 = fully opaque)
        margin: Margin from edge in pixels
    """

    enabled: bool = False
    logo_path: Path | None = None
    position: str = "top-center"
    height_percent: float = 0.15
    opacity: float = 1.0
    margin: int = 20


@dataclass
class PortraitConfig:
    """Configuration for portrait video conversion.

    Attributes:
        target_ratio: Target aspect ratio
        target_width: Target width in pixels (height calculated from ratio)
        crop_x_offset: Horizontal crop offset (0.0-1.0, 0.5 = center)
        crop_y_offset: Vertical crop offset (0.0-1.0, 0.5 = center)
        video_codec: Output video codec
        video_crf: Quality level (lower = better)
        video_preset: Encoding speed preset
        audio_codec: Output audio codec
        audio_bitrate: Audio bitrate
        faststart: Enable fast start for web playback
        logo: Logo overlay configuration
    """

    target_ratio: AspectRatio = AspectRatio.PORTRAIT_9_16
    target_width: int = 1080  # 1080x1920 for 9:16
    crop_x_offset: float = 0.5  # Center crop horizontally
    crop_y_offset: float = 0.5  # Center crop vertically
    video_codec: str = "libx264"
    video_crf: int = 23
    video_preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    faststart: bool = True
    logo: LogoOverlayConfig = field(default_factory=LogoOverlayConfig)

    @property
    def target_height(self) -> int:
        """Calculate target height from width and ratio."""
        w, h = self.target_ratio.value.split(":")
        return int(self.target_width * int(h) / int(w))

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get target dimensions as (width, height)."""
        return (self.target_width, self.target_height)


def calculate_crop_region(
    source_width: int,
    source_height: int,
    target_ratio: AspectRatio,
    x_offset: float = 0.5,
    y_offset: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Calculate crop region for target aspect ratio.

    Args:
        source_width: Source video width
        source_height: Source video height
        target_ratio: Target aspect ratio
        x_offset: Horizontal position (0.0=left, 0.5=center, 1.0=right)
        y_offset: Vertical position (0.0=top, 0.5=center, 1.0=bottom)

    Returns:
        Tuple of (crop_width, crop_height, x_position, y_position)
    """
    source_ratio = source_width / source_height
    target_ratio_value = target_ratio.width_ratio

    if source_ratio > target_ratio_value:
        # Source is wider than target - crop horizontally
        crop_height = source_height - (source_height % 2)  # Ensure even
        crop_width = int(crop_height * target_ratio_value)
        # Ensure even dimensions for encoding
        crop_width = crop_width - (crop_width % 2)

        max_x = source_width - crop_width
        x_pos = int(max_x * x_offset)
        y_pos = 0
    else:
        # Source is taller than target - crop vertically
        crop_width = source_width - (source_width % 2)  # Ensure even
        crop_height = int(crop_width / target_ratio_value)
        # Ensure even dimensions for encoding
        crop_height = crop_height - (crop_height % 2)

        x_pos = 0
        max_y = source_height - crop_height
        y_pos = int(max_y * y_offset)

    return (crop_width, crop_height, x_pos, y_pos)


def build_logo_overlay_filter(
    logo_config: LogoOverlayConfig,
    video_width: int,
    video_height: int,
) -> str | None:
    """Build FFmpeg filter for logo overlay.

    Args:
        logo_config: Logo overlay configuration
        video_width: Output video width
        video_height: Output video height

    Returns:
        FFmpeg overlay filter string, or None if logo disabled
    """
    if not logo_config.enabled or not logo_config.logo_path:
        return None

    logo_path = logo_config.logo_path
    if not logo_path.exists():
        return None

    # Calculate logo height in pixels
    logo_height = int(video_height * logo_config.height_percent)
    margin = logo_config.margin

    # Calculate position based on config
    position = logo_config.position.lower()

    # Vertical position
    if position.startswith("top"):
        y_pos = margin
    elif position.startswith("bottom"):
        y_pos = f"H-h-{margin}"
    else:
        y_pos = "(H-h)/2"

    # Horizontal position
    if position.endswith("left"):
        x_pos = margin
    elif position.endswith("right"):
        x_pos = f"W-w-{margin}"
    else:  # center
        x_pos = "(W-w)/2"

    # Escape path for FFmpeg (Windows paths need special handling)
    escaped_path = str(logo_path).replace("\\", "/").replace(":", "\\:")

    # Build the filter
    # Scale logo to desired height while maintaining aspect ratio
    # Then overlay at calculated position
    if logo_config.opacity < 1.0:
        # Apply alpha for transparency
        alpha_filter = f"format=rgba,colorchannelmixer=aa={logo_config.opacity}"
        filter_str = (
            f"[1:v]scale=-1:{logo_height},{alpha_filter}[logo];"
            f"[0:v][logo]overlay={x_pos}:{y_pos}"
        )
    else:
        filter_str = (
            f"[1:v]scale=-1:{logo_height}[logo];"
            f"[0:v][logo]overlay={x_pos}:{y_pos}"
        )

    return filter_str


class PortraitConverter:
    """Converts landscape videos to portrait format.

    Handles cropping, scaling, and encoding optimized for social media
    platforms like YouTube Shorts, TikTok, and Instagram Reels.

    Example usage:
        converter = PortraitConverter()
        converter.convert(
            input_path=Path("landscape.mp4"),
            output_path=Path("portrait.mp4"),
        )
    """

    def __init__(self, ffmpeg_path: str | None = None):
        """Initialize the converter.

        Args:
            ffmpeg_path: Path to FFmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
        self.ffmpeg_wrapper = FFmpegWrapper()

    def convert(
        self,
        input_path: Path | str,
        output_path: Path | str,
        config: PortraitConfig | None = None,
    ) -> Path:
        """Convert a video to portrait format.

        Args:
            input_path: Input video path
            output_path: Output video path
            config: Conversion configuration

        Returns:
            Path to output video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        config = config or PortraitConfig()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get source video info
        video_info = self.ffmpeg_wrapper.get_video_info(input_path)

        # Calculate crop region
        crop_w, crop_h, crop_x, crop_y = calculate_crop_region(
            source_width=video_info.width,
            source_height=video_info.height,
            target_ratio=config.target_ratio,
            x_offset=config.crop_x_offset,
            y_offset=config.crop_y_offset,
        )

        # Build filter string for crop and scale
        target_w, target_h = config.dimensions
        crop_scale_filter = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={target_w}:{target_h}:flags=lanczos"
        )

        # Check if we need logo overlay
        logo_config = config.logo
        use_logo = (
            logo_config.enabled
            and logo_config.logo_path
            and logo_config.logo_path.exists()
        )

        if use_logo:
            # Complex filter with logo overlay
            # Calculate logo dimensions
            logo_height = int(target_h * logo_config.height_percent)
            margin = logo_config.margin
            position = logo_config.position.lower()

            # Vertical position
            if position.startswith("top"):
                y_pos = margin
            elif position.startswith("bottom"):
                y_pos = f"H-h-{margin}"
            else:
                y_pos = "(H-h)/2"

            # Horizontal position
            if position.endswith("left"):
                x_pos = margin
            elif position.endswith("right"):
                x_pos = f"W-w-{margin}"
            else:  # center
                x_pos = "(W-w)/2"

            # Build filter complex
            if logo_config.opacity < 1.0:
                alpha_filter = f"format=rgba,colorchannelmixer=aa={logo_config.opacity}"
                filter_complex = (
                    f"[0:v]{crop_scale_filter}[cropped];"
                    f"[1:v]scale=-1:{logo_height},{alpha_filter}[logo];"
                    f"[cropped][logo]overlay={x_pos}:{y_pos}[out]"
                )
            else:
                filter_complex = (
                    f"[0:v]{crop_scale_filter}[cropped];"
                    f"[1:v]scale=-1:{logo_height}[logo];"
                    f"[cropped][logo]overlay={x_pos}:{y_pos}[out]"
                )

            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-i", str(logo_config.logo_path),
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-map", "0:a",
                "-c:v", config.video_codec,
                "-crf", str(config.video_crf),
                "-preset", config.video_preset,
                "-c:a", config.audio_codec,
                "-b:a", config.audio_bitrate,
            ]
        else:
            # Simple filter without logo
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-vf", crop_scale_filter,
                "-c:v", config.video_codec,
                "-crf", str(config.video_crf),
                "-preset", config.video_preset,
                "-c:a", config.audio_codec,
                "-b:a", config.audio_bitrate,
            ]

        if config.faststart:
            cmd.extend(["-movflags", "+faststart"])

        cmd.extend(["-y", str(output_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        return output_path

    def convert_with_captions(
        self,
        input_path: Path | str,
        output_path: Path | str,
        caption_track: "CaptionTrack",
        config: PortraitConfig | None = None,
        caption_style: "CaptionStyle" | None = None,
    ) -> Path:
        """Convert to portrait with burned-in captions and optional logo.

        Args:
            input_path: Input video path
            output_path: Output video path
            caption_track: Captions to burn in
            config: Conversion configuration
            caption_style: Caption styling

        Returns:
            Path to output video
        """
        from clip_video.captions.renderer import CaptionTrack, CaptionRenderer
        from clip_video.captions.styles import CaptionStyle, YOUTUBE_SHORTS_STYLE

        input_path = Path(input_path)
        output_path = Path(output_path)
        config = config or PortraitConfig()
        caption_style = caption_style or YOUTUBE_SHORTS_STYLE

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get source video info
        video_info = self.ffmpeg_wrapper.get_video_info(input_path)

        # Calculate crop region
        crop_w, crop_h, crop_x, crop_y = calculate_crop_region(
            source_width=video_info.width,
            source_height=video_info.height,
            target_ratio=config.target_ratio,
            x_offset=config.crop_x_offset,
            y_offset=config.crop_y_offset,
        )

        # Build filter parts for crop, scale, and captions
        target_w, target_h = config.dimensions
        filter_parts = [
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
            f"scale={target_w}:{target_h}:flags=lanczos",
        ]

        # Add caption drawtext filters
        for caption in caption_track.captions:
            cap_style = caption.style or caption_style
            params = cap_style.to_drawtext_params(target_w, target_h)

            # Escape text
            text = caption.text
            if cap_style.uppercase:
                text = text.upper()
            text = text.replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")

            param_str = ":".join(f"{k}={v}" for k, v in params.items())
            enable = f"between(t,{caption.start_time},{caption.end_time})"

            filter_parts.append(
                f"drawtext=text='{text}':{param_str}:enable='{enable}'"
            )

        # Check if we need logo overlay
        logo_config = config.logo
        use_logo = (
            logo_config.enabled
            and logo_config.logo_path
            and logo_config.logo_path.exists()
        )

        if use_logo:
            # Build complex filter with logo overlay
            # First apply crop, scale, and captions to video stream
            video_filter = ",".join(filter_parts)

            # Calculate logo dimensions
            logo_height = int(target_h * logo_config.height_percent)
            margin = logo_config.margin
            position = logo_config.position.lower()

            # Vertical position
            if position.startswith("top"):
                y_pos = margin
            elif position.startswith("bottom"):
                y_pos = f"H-h-{margin}"
            else:
                y_pos = "(H-h)/2"

            # Horizontal position
            if position.endswith("left"):
                x_pos = margin
            elif position.endswith("right"):
                x_pos = f"W-w-{margin}"
            else:  # center
                x_pos = "(W-w)/2"

            # Build complex filter
            if logo_config.opacity < 1.0:
                alpha_filter = f"format=rgba,colorchannelmixer=aa={logo_config.opacity}"
                filter_complex = (
                    f"[0:v]{video_filter}[captioned];"
                    f"[1:v]scale=-1:{logo_height},{alpha_filter}[logo];"
                    f"[captioned][logo]overlay={x_pos}:{y_pos}[out]"
                )
            else:
                filter_complex = (
                    f"[0:v]{video_filter}[captioned];"
                    f"[1:v]scale=-1:{logo_height}[logo];"
                    f"[captioned][logo]overlay={x_pos}:{y_pos}[out]"
                )

            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-i", str(logo_config.logo_path),
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-map", "0:a",
                "-c:v", config.video_codec,
                "-crf", str(config.video_crf),
                "-preset", config.video_preset,
                "-c:a", config.audio_codec,
                "-b:a", config.audio_bitrate,
            ]
        else:
            # Simple filter without logo
            filter_str = ",".join(filter_parts)

            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path),
                "-vf", filter_str,
                "-c:v", config.video_codec,
                "-crf", str(config.video_crf),
                "-preset", config.video_preset,
                "-c:a", config.audio_codec,
                "-b:a", config.audio_bitrate,
            ]

        if config.faststart:
            cmd.extend(["-movflags", "+faststart"])

        cmd.extend(["-y", str(output_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        return output_path

    def get_optimal_crop(
        self,
        input_path: Path | str,
        target_ratio: AspectRatio = AspectRatio.PORTRAIT_9_16,
    ) -> Tuple[int, int, int, int]:
        """Calculate optimal crop region for a video.

        Uses center crop by default. Returns crop parameters that can
        be adjusted before conversion.

        Args:
            input_path: Input video path
            target_ratio: Target aspect ratio

        Returns:
            Tuple of (crop_width, crop_height, x_position, y_position)
        """
        video_info = self.ffmpeg_wrapper.get_video_info(input_path)

        return calculate_crop_region(
            source_width=video_info.width,
            source_height=video_info.height,
            target_ratio=target_ratio,
        )


# Platform-specific configurations
YOUTUBE_SHORTS_CONFIG = PortraitConfig(
    target_ratio=AspectRatio.PORTRAIT_9_16,
    target_width=1080,
    crop_x_offset=0.5,  # Center crop by default - override via brand config
    video_crf=20,
    video_preset="slow",
)

TIKTOK_CONFIG = PortraitConfig(
    target_ratio=AspectRatio.PORTRAIT_9_16,
    target_width=1080,
    video_crf=18,  # Higher quality for TikTok
    video_preset="medium",
)

INSTAGRAM_REELS_CONFIG = PortraitConfig(
    target_ratio=AspectRatio.PORTRAIT_9_16,
    target_width=1080,
    video_crf=20,
    faststart=True,
)

INSTAGRAM_FEED_CONFIG = PortraitConfig(
    target_ratio=AspectRatio.PORTRAIT_4_5,
    target_width=1080,
    video_crf=20,
)

LINKEDIN_CONFIG = PortraitConfig(
    target_ratio=AspectRatio.SQUARE_1_1,
    target_width=1080,
    video_crf=22,
)
