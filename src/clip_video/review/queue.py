"""Review queue for rejected clips.

Provides file-based storage for clips that failed validation,
allowing human review and potential override.

D022: File-based storage in review/ folder
D023: Each entry includes rejection reasons, transcript segment, preview command
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RejectedClip:
    """A clip that failed validation and needs human review.

    Attributes:
        clip_id: Unique identifier for the clip
        video_path: Path to the source video
        start_time: Clip start time in seconds
        end_time: Clip end time in seconds
        transcript_segment: The transcript text for this segment
        rejection_reasons: List of reasons the clip was rejected
        validation_details: Detailed validation results
        rejected_at: ISO timestamp of when clip was rejected
        replacement_attempts: Number of times replacement was attempted
    """

    clip_id: str
    video_path: str
    start_time: float
    end_time: float
    transcript_segment: str
    rejection_reasons: list[str]
    validation_details: dict[str, Any] = field(default_factory=dict)
    rejected_at: str = ""
    replacement_attempts: int = 0

    def __post_init__(self):
        """Set rejected_at timestamp if not provided."""
        if not self.rejected_at:
            self.rejected_at = datetime.now().isoformat()

    @property
    def duration(self) -> float:
        """Calculate clip duration in seconds."""
        return self.end_time - self.start_time

    @property
    def preview_command(self) -> str:
        """Generate CLI command to preview this clip.

        Returns a command that can be run to quickly preview the
        rejected clip without regenerating it.
        """
        return (
            f'clip-video preview "{self.video_path}" '
            f'--start {self.start_time:.1f} --end {self.end_time:.1f}'
        )

    @property
    def ffplay_command(self) -> str:
        """Generate ffplay command for quick preview.

        This uses ffplay directly for systems with FFmpeg installed.
        """
        return (
            f'ffplay -ss {self.start_time:.1f} -t {self.duration:.1f} '
            f'-autoexit "{self.video_path}"'
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "transcript_segment": self.transcript_segment,
            "rejection_reasons": self.rejection_reasons,
            "validation_details": self.validation_details,
            "rejected_at": self.rejected_at,
            "replacement_attempts": self.replacement_attempts,
            "preview_command": self.preview_command,
            "ffplay_command": self.ffplay_command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RejectedClip":
        """Create from dictionary."""
        return cls(
            clip_id=data["clip_id"],
            video_path=data["video_path"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            transcript_segment=data["transcript_segment"],
            rejection_reasons=data["rejection_reasons"],
            validation_details=data.get("validation_details", {}),
            rejected_at=data.get("rejected_at", ""),
            replacement_attempts=data.get("replacement_attempts", 0),
        )


class ReviewQueue:
    """Manages the review queue for rejected clips.

    Provides file-based storage where each rejected clip is saved
    as a separate JSON file in the review directory.

    Attributes:
        review_dir: Path to the review directory
    """

    def __init__(self, review_dir: Path | str):
        """Initialize the review queue.

        Args:
            review_dir: Path to directory for storing review files
        """
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def add(self, clip: RejectedClip) -> Path:
        """Add a rejected clip to the queue.

        Args:
            clip: The rejected clip to add

        Returns:
            Path to the saved JSON file
        """
        # Create filename: clip_id_timestamp.json
        # Sanitize timestamp for filename (replace : with -)
        safe_timestamp = clip.rejected_at.replace(":", "-").replace(".", "-")
        filename = f"{clip.clip_id}_{safe_timestamp}.json"
        filepath = self.review_dir / filename

        # Write with retry logic for Windows/Dropbox compatibility
        self._write_json(filepath, clip.to_dict())

        return filepath

    def _write_json(self, filepath: Path, data: dict) -> None:
        """Write JSON with retry logic for file locking issues.

        Args:
            filepath: Path to write to
            data: Dictionary to serialize
        """
        temp_path = filepath.with_suffix(".tmp")
        max_retries = 5

        for attempt in range(max_retries):
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # On Windows, may need to remove target first
                if os.name == "nt" and filepath.exists():
                    try:
                        filepath.unlink()
                    except PermissionError:
                        pass

                temp_path.replace(filepath)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    # Last resort: direct write
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    try:
                        temp_path.unlink()
                    except (PermissionError, FileNotFoundError):
                        pass

    def get(self, clip_id: str) -> RejectedClip | None:
        """Get a specific rejected clip by ID.

        Args:
            clip_id: The clip ID to find

        Returns:
            RejectedClip if found, None otherwise
        """
        for filepath in self.review_dir.glob(f"{clip_id}_*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return RejectedClip.from_dict(data)
        return None

    def remove(self, clip_id: str) -> bool:
        """Remove a clip from the review queue.

        Args:
            clip_id: The clip ID to remove

        Returns:
            True if clip was removed, False if not found
        """
        for filepath in self.review_dir.glob(f"{clip_id}_*.json"):
            filepath.unlink()
            return True
        return False

    def list_all(self) -> list[RejectedClip]:
        """List all clips in the review queue.

        Returns:
            List of RejectedClip objects, sorted by rejection time (newest first)
        """
        clips = []
        for filepath in self.review_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    clips.append(RejectedClip.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed files
                continue

        return sorted(clips, key=lambda c: c.rejected_at, reverse=True)

    def list_by_video(self, video_path: str) -> list[RejectedClip]:
        """List clips from a specific video.

        Args:
            video_path: Path to the source video

        Returns:
            List of RejectedClip objects from this video
        """
        all_clips = self.list_all()
        return [c for c in all_clips if c.video_path == video_path]

    def count(self) -> int:
        """Get count of clips in review queue.

        Returns:
            Number of clips in queue
        """
        return len(list(self.review_dir.glob("*.json")))

    def get_summary(self) -> dict[str, Any]:
        """Get summary of review queue.

        Returns:
            Dictionary with queue statistics
        """
        clips = self.list_all()
        return {
            "total_clips": len(clips),
            "by_reason": self._group_by_reason(clips),
            "by_video": self._group_by_video(clips),
            "total_duration": sum(c.duration for c in clips),
        }

    def _group_by_reason(self, clips: list[RejectedClip]) -> dict[str, int]:
        """Group clips by rejection reason.

        Args:
            clips: List of rejected clips

        Returns:
            Dictionary mapping reason to count
        """
        reasons: dict[str, int] = {}
        for clip in clips:
            for reason in clip.rejection_reasons:
                # Normalize reason for grouping
                key = reason.split(":")[0].strip() if ":" in reason else reason
                reasons[key] = reasons.get(key, 0) + 1
        return reasons

    def _group_by_video(self, clips: list[RejectedClip]) -> dict[str, int]:
        """Group clips by source video.

        Args:
            clips: List of rejected clips

        Returns:
            Dictionary mapping video path to count
        """
        videos: dict[str, int] = {}
        for clip in clips:
            video_name = Path(clip.video_path).name
            videos[video_name] = videos.get(video_name, 0) + 1
        return videos

    def clear(self) -> int:
        """Clear all clips from the review queue.

        Returns:
            Number of clips removed
        """
        count = 0
        for filepath in self.review_dir.glob("*.json"):
            filepath.unlink()
            count += 1
        return count
