"""FFmpeg binary management for clip-video.

Handles embedded FFmpeg binaries to eliminate external dependency issues.
Uses imageio-ffmpeg as the primary approach, with fallback options for
downloading platform-specific binaries.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field


class FFmpegInfo(NamedTuple):
    """Information about FFmpeg installation."""

    path: str
    version: str
    available: bool
    source: str  # "imageio", "custom", "system", or "not_found"


class FFmpegConfig(BaseModel):
    """Configuration for FFmpeg binary location.

    Allows advanced users to specify custom binary locations.
    """

    custom_ffmpeg_path: str | None = Field(
        default=None,
        description="Custom path to FFmpeg executable"
    )
    custom_ffprobe_path: str | None = Field(
        default=None,
        description="Custom path to FFprobe executable"
    )
    prefer_system: bool = Field(
        default=False,
        description="Prefer system FFmpeg over bundled version"
    )


# Default location for custom binaries within the package
def get_bin_directory() -> Path:
    """Get the directory for storing custom FFmpeg binaries.

    Returns:
        Path to the bin directory within the package.
    """
    return Path(__file__).parent / "bin"


def ensure_bin_directory() -> Path:
    """Ensure the bin directory exists.

    Returns:
        Path to the bin directory.
    """
    bin_dir = get_bin_directory()
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir


def _get_ffmpeg_from_imageio() -> str | None:
    """Get FFmpeg path from imageio-ffmpeg package.

    Returns:
        Path to FFmpeg executable, or None if not available.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _get_ffprobe_from_imageio() -> str | None:
    """Get FFprobe path from imageio-ffmpeg package.

    Note: imageio-ffmpeg does not bundle ffprobe, so we look for it
    next to the ffmpeg executable.

    Returns:
        Path to FFprobe executable, or None if not available.
    """
    ffmpeg_path = _get_ffmpeg_from_imageio()
    if ffmpeg_path is None:
        return None

    ffmpeg_dir = Path(ffmpeg_path).parent

    # Try common names for ffprobe
    if platform.system() == "Windows":
        ffprobe_names = ["ffprobe.exe", "ffprobe"]
    else:
        ffprobe_names = ["ffprobe"]

    for name in ffprobe_names:
        ffprobe_path = ffmpeg_dir / name
        if ffprobe_path.exists():
            return str(ffprobe_path)

    return None


def _get_custom_ffmpeg() -> str | None:
    """Get FFmpeg from the custom bin directory.

    Returns:
        Path to FFmpeg executable, or None if not available.
    """
    bin_dir = get_bin_directory()

    if platform.system() == "Windows":
        ffmpeg_path = bin_dir / "ffmpeg.exe"
    else:
        ffmpeg_path = bin_dir / "ffmpeg"

    if ffmpeg_path.exists():
        return str(ffmpeg_path)
    return None


def _get_custom_ffprobe() -> str | None:
    """Get FFprobe from the custom bin directory.

    Returns:
        Path to FFprobe executable, or None if not available.
    """
    bin_dir = get_bin_directory()

    if platform.system() == "Windows":
        ffprobe_path = bin_dir / "ffprobe.exe"
    else:
        ffprobe_path = bin_dir / "ffprobe"

    if ffprobe_path.exists():
        return str(ffprobe_path)
    return None


def _get_system_ffmpeg() -> str | None:
    """Get FFmpeg from system PATH.

    Returns:
        Path to FFmpeg executable, or None if not available.
    """
    return shutil.which("ffmpeg")


def _get_system_ffprobe() -> str | None:
    """Get FFprobe from system PATH.

    Returns:
        Path to FFprobe executable, or None if not available.
    """
    return shutil.which("ffprobe")


def _get_ffmpeg_version(ffmpeg_path: str) -> str | None:
    """Get version string from FFmpeg executable.

    Args:
        ffmpeg_path: Path to FFmpeg executable.

    Returns:
        Version string, or None if unable to determine.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        if result.returncode == 0:
            # First line typically contains version info
            first_line = result.stdout.split("\n")[0]
            # Extract version number (e.g., "ffmpeg version 6.0-full_build-www.gyan.dev")
            if "version" in first_line.lower():
                parts = first_line.split("version")
                if len(parts) > 1:
                    version_part = parts[1].strip().split()[0]
                    return version_part
            return first_line.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_ffmpeg_path(config: FFmpegConfig | None = None) -> str | None:
    """Get the path to FFmpeg executable.

    Searches in the following order (unless config specifies otherwise):
    1. Custom path from config
    2. Custom bin directory
    3. imageio-ffmpeg bundled binary
    4. System PATH

    Args:
        config: Optional configuration for custom paths.

    Returns:
        Path to FFmpeg executable, or None if not found.
    """
    if config is None:
        config = FFmpegConfig()

    # Check custom path first
    if config.custom_ffmpeg_path:
        if Path(config.custom_ffmpeg_path).exists():
            return config.custom_ffmpeg_path

    # If prefer_system is set, check system first
    if config.prefer_system:
        system_path = _get_system_ffmpeg()
        if system_path:
            return system_path

    # Check custom bin directory
    custom_path = _get_custom_ffmpeg()
    if custom_path:
        return custom_path

    # Check imageio-ffmpeg
    imageio_path = _get_ffmpeg_from_imageio()
    if imageio_path:
        return imageio_path

    # Fall back to system PATH
    return _get_system_ffmpeg()


def get_ffprobe_path(config: FFmpegConfig | None = None) -> str | None:
    """Get the path to FFprobe executable.

    Searches in the following order (unless config specifies otherwise):
    1. Custom path from config
    2. Custom bin directory
    3. Next to imageio-ffmpeg's FFmpeg
    4. System PATH

    Args:
        config: Optional configuration for custom paths.

    Returns:
        Path to FFprobe executable, or None if not found.
    """
    if config is None:
        config = FFmpegConfig()

    # Check custom path first
    if config.custom_ffprobe_path:
        if Path(config.custom_ffprobe_path).exists():
            return config.custom_ffprobe_path

    # If prefer_system is set, check system first
    if config.prefer_system:
        system_path = _get_system_ffprobe()
        if system_path:
            return system_path

    # Check custom bin directory
    custom_path = _get_custom_ffprobe()
    if custom_path:
        return custom_path

    # Check next to imageio-ffmpeg's FFmpeg
    imageio_path = _get_ffprobe_from_imageio()
    if imageio_path:
        return imageio_path

    # Fall back to system PATH
    return _get_system_ffprobe()


def get_ffmpeg_info(config: FFmpegConfig | None = None) -> FFmpegInfo:
    """Get comprehensive information about FFmpeg installation.

    Args:
        config: Optional configuration for custom paths.

    Returns:
        FFmpegInfo with path, version, availability, and source.
    """
    if config is None:
        config = FFmpegConfig()

    # Determine source and path
    source = "not_found"
    path = None

    # Check custom path first
    if config.custom_ffmpeg_path and Path(config.custom_ffmpeg_path).exists():
        path = config.custom_ffmpeg_path
        source = "custom"
    elif config.prefer_system and _get_system_ffmpeg():
        path = _get_system_ffmpeg()
        source = "system"
    elif _get_custom_ffmpeg():
        path = _get_custom_ffmpeg()
        source = "custom"
    elif _get_ffmpeg_from_imageio():
        path = _get_ffmpeg_from_imageio()
        source = "imageio"
    elif _get_system_ffmpeg():
        path = _get_system_ffmpeg()
        source = "system"

    if path is None:
        return FFmpegInfo(
            path="",
            version="",
            available=False,
            source="not_found"
        )

    version = _get_ffmpeg_version(path) or "unknown"

    return FFmpegInfo(
        path=path,
        version=version,
        available=True,
        source=source
    )


def verify_ffmpeg(config: FFmpegConfig | None = None) -> tuple[bool, str]:
    """Verify FFmpeg is available and working.

    Args:
        config: Optional configuration for custom paths.

    Returns:
        Tuple of (success, message).
    """
    info = get_ffmpeg_info(config)

    if not info.available:
        return (False, "FFmpeg not found. Please install imageio-ffmpeg or add FFmpeg to PATH.")

    # Test that FFmpeg actually works
    try:
        result = subprocess.run(
            [info.path, "-version"],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        if result.returncode != 0:
            return (False, f"FFmpeg found at {info.path} but returned error code {result.returncode}")
    except subprocess.TimeoutExpired:
        return (False, f"FFmpeg at {info.path} timed out during verification")
    except OSError as e:
        return (False, f"Failed to run FFmpeg at {info.path}: {e}")

    return (True, f"FFmpeg {info.version} available ({info.source}): {info.path}")


def check_ffprobe(config: FFmpegConfig | None = None) -> tuple[bool, str]:
    """Check if FFprobe is available.

    Args:
        config: Optional configuration for custom paths.

    Returns:
        Tuple of (available, message).
    """
    path = get_ffprobe_path(config)

    if path is None:
        return (False, "FFprobe not found")

    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        if result.returncode == 0:
            # Get version from output
            first_line = result.stdout.decode().split("\n")[0]
            return (True, f"FFprobe available: {path}")
        return (False, f"FFprobe at {path} returned error code {result.returncode}")
    except subprocess.TimeoutExpired:
        return (False, f"FFprobe at {path} timed out")
    except OSError as e:
        return (False, f"Failed to run FFprobe at {path}: {e}")


def install_custom_ffmpeg(ffmpeg_path: str, ffprobe_path: str | None = None) -> tuple[bool, str]:
    """Install custom FFmpeg binaries to the package bin directory.

    Allows users to provide their own FFmpeg binaries without modifying
    system PATH or requiring admin privileges.

    Args:
        ffmpeg_path: Path to FFmpeg executable to install.
        ffprobe_path: Optional path to FFprobe executable.

    Returns:
        Tuple of (success, message).
    """
    bin_dir = ensure_bin_directory()

    ffmpeg_src = Path(ffmpeg_path)
    if not ffmpeg_src.exists():
        return (False, f"Source FFmpeg not found: {ffmpeg_path}")

    # Determine destination filename
    if platform.system() == "Windows":
        ffmpeg_dest = bin_dir / "ffmpeg.exe"
    else:
        ffmpeg_dest = bin_dir / "ffmpeg"

    try:
        shutil.copy2(ffmpeg_src, ffmpeg_dest)
        # Make executable on Unix
        if platform.system() != "Windows":
            ffmpeg_dest.chmod(0o755)
    except OSError as e:
        return (False, f"Failed to copy FFmpeg: {e}")

    message = f"FFmpeg installed to {ffmpeg_dest}"

    # Copy FFprobe if provided
    if ffprobe_path:
        ffprobe_src = Path(ffprobe_path)
        if ffprobe_src.exists():
            if platform.system() == "Windows":
                ffprobe_dest = bin_dir / "ffprobe.exe"
            else:
                ffprobe_dest = bin_dir / "ffprobe"

            try:
                shutil.copy2(ffprobe_src, ffprobe_dest)
                if platform.system() != "Windows":
                    ffprobe_dest.chmod(0o755)
                message += f"\nFFprobe installed to {ffprobe_dest}"
            except OSError as e:
                message += f"\nWarning: Failed to copy FFprobe: {e}"

    return (True, message)


def get_dependency_report() -> dict[str, dict[str, str | bool]]:
    """Generate a comprehensive dependency report.

    Returns:
        Dictionary with dependency status information.
    """
    ffmpeg_info = get_ffmpeg_info()
    ffprobe_available, ffprobe_msg = check_ffprobe()

    report = {
        "ffmpeg": {
            "available": ffmpeg_info.available,
            "path": ffmpeg_info.path,
            "version": ffmpeg_info.version,
            "source": ffmpeg_info.source,
        },
        "ffprobe": {
            "available": ffprobe_available,
            "path": get_ffprobe_path() or "",
            "message": ffprobe_msg,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
        },
    }

    # Check imageio-ffmpeg availability
    try:
        import imageio_ffmpeg
        report["imageio_ffmpeg"] = {
            "available": True,
            "version": getattr(imageio_ffmpeg, "__version__", "unknown"),
        }
    except ImportError:
        report["imageio_ffmpeg"] = {
            "available": False,
            "version": "",
        }

    return report
