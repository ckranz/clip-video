"""Import and export functions for VTT and SRT subtitle formats.

Supports:
- Importing VTT/SRT files to Transcript objects
- Exporting Transcript objects to VTT/SRT formats
- Preserving timing information during conversion
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from clip_video.models.transcript import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


class SubtitleParseError(Exception):
    """Raised when parsing a subtitle file fails."""

    pass


def _parse_vtt_timestamp(timestamp: str) -> float:
    """Parse a VTT timestamp to seconds.

    VTT format: HH:MM:SS.mmm or MM:SS.mmm

    Args:
        timestamp: VTT timestamp string

    Returns:
        Time in seconds

    Raises:
        SubtitleParseError: If timestamp format is invalid
    """
    timestamp = timestamp.strip()

    # Try HH:MM:SS.mmm format
    match = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})", timestamp)
    if match:
        hours, minutes, seconds, millis = match.groups()
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000
        )

    # Try MM:SS.mmm format
    match = re.match(r"(\d{1,2}):(\d{2})\.(\d{3})", timestamp)
    if match:
        minutes, seconds, millis = match.groups()
        return int(minutes) * 60 + int(seconds) + int(millis) / 1000

    raise SubtitleParseError(f"Invalid VTT timestamp: {timestamp}")


def _parse_srt_timestamp(timestamp: str) -> float:
    """Parse an SRT timestamp to seconds.

    SRT format: HH:MM:SS,mmm

    Args:
        timestamp: SRT timestamp string

    Returns:
        Time in seconds

    Raises:
        SubtitleParseError: If timestamp format is invalid
    """
    timestamp = timestamp.strip()

    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", timestamp)
    if match:
        hours, minutes, seconds, millis = match.groups()
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000
        )

    raise SubtitleParseError(f"Invalid SRT timestamp: {timestamp}")


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds to VTT timestamp.

    Args:
        seconds: Time in seconds

    Returns:
        VTT timestamp string (HH:MM:SS.mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp.

    Args:
        seconds: Time in seconds

    Returns:
        SRT timestamp string (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_text_to_words(
    text: str, start: float, end: float
) -> list[TranscriptWord]:
    """Split text into words with estimated timestamps.

    When word-level timing is not available, estimates word timestamps
    based on character count.

    Args:
        text: Text to split
        start: Segment start time
        end: Segment end time

    Returns:
        List of TranscriptWord objects with estimated timing
    """
    words_text = text.split()
    if not words_text:
        return []

    # Calculate duration per character (excluding spaces)
    total_chars = sum(len(w) for w in words_text)
    if total_chars == 0:
        return []

    duration = end - start
    time_per_char = duration / total_chars

    words = []
    current_time = start

    for word_text in words_text:
        word_duration = len(word_text) * time_per_char
        word_end = current_time + word_duration

        words.append(
            TranscriptWord(
                word=word_text,
                start=round(current_time, 3),
                end=round(word_end, 3),
                confidence=1.0,  # Unknown confidence for imported subtitles
            )
        )
        current_time = word_end

    return words


def import_vtt(
    content: str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import a VTT (WebVTT) subtitle file to a Transcript.

    Parses VTT format and creates segments with estimated word-level timing.

    Args:
        content: VTT file content as string
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object

    Raises:
        SubtitleParseError: If VTT format is invalid
    """
    lines = content.strip().split("\n")

    # Check for WEBVTT header
    if not lines or not lines[0].strip().startswith("WEBVTT"):
        raise SubtitleParseError("Invalid VTT file: missing WEBVTT header")

    segments: list[TranscriptSegment] = []
    segment_id = 0

    i = 1  # Skip WEBVTT header
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and metadata
        if not line or line.startswith("NOTE") or line.startswith("STYLE"):
            i += 1
            continue

        # Skip cue identifiers (optional in VTT)
        if not "-->" in line:
            # Check if next line has timestamp
            if i + 1 < len(lines) and "-->" in lines[i + 1]:
                i += 1
                line = lines[i].strip()
            else:
                i += 1
                continue

        # Parse timestamp line
        timestamp_match = re.match(
            r"([0-9:.]+)\s*-->\s*([0-9:.]+)", line
        )
        if not timestamp_match:
            i += 1
            continue

        start_str, end_str = timestamp_match.groups()
        try:
            start = _parse_vtt_timestamp(start_str)
            end = _parse_vtt_timestamp(end_str)
        except SubtitleParseError:
            i += 1
            continue

        # Collect text lines until empty line or end
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            # Remove VTT tags like <c> </c> <b> </b>
            text = re.sub(r"<[^>]+>", "", lines[i])
            text_lines.append(text.strip())
            i += 1

        text = " ".join(text_lines)
        if not text:
            continue

        # Create segment with estimated word timing
        words = _split_text_to_words(text, start, end)

        segments.append(
            TranscriptSegment(
                id=segment_id,
                text=text,
                start=start,
                end=end,
                words=words,
            )
        )
        segment_id += 1

    # Calculate full text and duration
    full_text = " ".join(seg.text for seg in segments)
    duration = segments[-1].end if segments else 0.0

    return Transcript(
        video_path=video_path,
        language=language,
        created_at=datetime.now(),
        provider="vtt_import",
        model="",
        duration=duration,
        segments=segments,
        full_text=full_text,
    )


def import_srt(
    content: str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import an SRT subtitle file to a Transcript.

    Parses SRT format and creates segments with estimated word-level timing.

    Args:
        content: SRT file content as string
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object

    Raises:
        SubtitleParseError: If SRT format is invalid
    """
    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Split into cue blocks (separated by blank lines)
    blocks = re.split(r"\n\n+", content.strip())

    segments: list[TranscriptSegment] = []
    segment_id = 0

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # First line should be cue number (or timestamp if no number)
        line_idx = 0

        # Check if first line is a number
        if lines[line_idx].strip().isdigit():
            line_idx += 1

        if line_idx >= len(lines):
            continue

        # Parse timestamp line
        timestamp_line = lines[line_idx]
        timestamp_match = re.match(
            r"([0-9:,]+)\s*-->\s*([0-9:,]+)", timestamp_line
        )
        if not timestamp_match:
            continue

        start_str, end_str = timestamp_match.groups()
        try:
            start = _parse_srt_timestamp(start_str)
            end = _parse_srt_timestamp(end_str)
        except SubtitleParseError:
            continue

        # Remaining lines are text
        line_idx += 1
        text_lines = []
        while line_idx < len(lines):
            # Remove HTML-style tags
            text = re.sub(r"<[^>]+>", "", lines[line_idx])
            text_lines.append(text.strip())
            line_idx += 1

        text = " ".join(text_lines)
        if not text:
            continue

        # Create segment with estimated word timing
        words = _split_text_to_words(text, start, end)

        segments.append(
            TranscriptSegment(
                id=segment_id,
                text=text,
                start=start,
                end=end,
                words=words,
            )
        )
        segment_id += 1

    # Calculate full text and duration
    full_text = " ".join(seg.text for seg in segments)
    duration = segments[-1].end if segments else 0.0

    return Transcript(
        video_path=video_path,
        language=language,
        created_at=datetime.now(),
        provider="srt_import",
        model="",
        duration=duration,
        segments=segments,
        full_text=full_text,
    )


def import_vtt_file(
    path: Path | str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import a VTT file from disk.

    Args:
        path: Path to the VTT file
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return import_vtt(content, video_path, language)


def import_srt_file(
    path: Path | str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import an SRT file from disk.

    Args:
        path: Path to the SRT file
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return import_srt(content, video_path, language)


def export_vtt(transcript: Transcript) -> str:
    """Export a Transcript to VTT format.

    Args:
        transcript: Transcript to export

    Returns:
        VTT file content as string
    """
    lines = ["WEBVTT", ""]

    for segment in transcript.segments:
        # Timestamp line
        start = _format_vtt_timestamp(segment.start)
        end = _format_vtt_timestamp(segment.end)
        lines.append(f"{start} --> {end}")

        # Text line
        lines.append(segment.text)
        lines.append("")

    return "\n".join(lines)


def export_srt(transcript: Transcript) -> str:
    """Export a Transcript to SRT format.

    Args:
        transcript: Transcript to export

    Returns:
        SRT file content as string
    """
    blocks = []

    for i, segment in enumerate(transcript.segments, 1):
        # Cue number
        block_lines = [str(i)]

        # Timestamp line
        start = _format_srt_timestamp(segment.start)
        end = _format_srt_timestamp(segment.end)
        block_lines.append(f"{start} --> {end}")

        # Text line
        block_lines.append(segment.text)

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks) + "\n"


def export_vtt_file(transcript: Transcript, path: Path | str) -> Path:
    """Export a Transcript to a VTT file.

    Args:
        transcript: Transcript to export
        path: Path to save the VTT file

    Returns:
        Path to the saved file
    """
    path = Path(path)
    content = export_vtt(transcript)
    path.write_text(content, encoding="utf-8")
    return path


def export_srt_file(transcript: Transcript, path: Path | str) -> Path:
    """Export a Transcript to an SRT file.

    Args:
        transcript: Transcript to export
        path: Path to save the SRT file

    Returns:
        Path to the saved file
    """
    path = Path(path)
    content = export_srt(transcript)
    path.write_text(content, encoding="utf-8")
    return path


def detect_subtitle_format(content: str) -> str:
    """Detect the format of a subtitle file.

    Args:
        content: File content as string

    Returns:
        Format string: "vtt", "srt", or "unknown"
    """
    content = content.strip()

    if content.startswith("WEBVTT"):
        return "vtt"

    # Check for SRT format (starts with number, then timestamp)
    lines = content.split("\n")
    if len(lines) >= 2:
        if lines[0].strip().isdigit():
            if "-->" in lines[1] and "," in lines[1]:
                return "srt"

    # Check for timestamp patterns
    if re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}", content):
        return "srt"
    if re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}", content):
        return "vtt"

    return "unknown"


def import_subtitle(
    content: str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import a subtitle file with auto-detection of format.

    Args:
        content: File content as string
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object

    Raises:
        SubtitleParseError: If format cannot be detected or is invalid
    """
    format_type = detect_subtitle_format(content)

    if format_type == "vtt":
        return import_vtt(content, video_path, language)
    elif format_type == "srt":
        return import_srt(content, video_path, language)
    else:
        raise SubtitleParseError("Unable to detect subtitle format")


def import_subtitle_file(
    path: Path | str,
    video_path: str = "",
    language: str = "en",
) -> Transcript:
    """Import a subtitle file from disk with auto-detection.

    Args:
        path: Path to the subtitle file
        video_path: Path to the source video (for metadata)
        language: Language code

    Returns:
        Transcript object
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return import_subtitle(content, video_path, language)
