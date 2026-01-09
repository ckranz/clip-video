"""Transcription progress tracking for resume capability.

Tracks which videos have been transcribed to enable resuming after
interruption without re-processing completed files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TranscriptionStatus(str, Enum):
    """Status of a video transcription."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VideoProgress:
    """Progress information for a single video."""

    video_path: str
    status: TranscriptionStatus = TranscriptionStatus.PENDING
    transcript_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None
    cost_usd: float | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "status": self.status.value,
            "transcript_path": self.transcript_path,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "cost_usd": self.cost_usd,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoProgress":
        """Create from dictionary."""
        return cls(
            video_path=data["video_path"],
            status=TranscriptionStatus(data.get("status", "pending")),
            transcript_path=data.get("transcript_path"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds", 0.0),
            error_message=data.get("error_message"),
            cost_usd=data.get("cost_usd"),
            attempts=data.get("attempts", 0),
        )


@dataclass
class TranscriptionProgress:
    """Tracks transcription progress for a brand.

    Provides resume capability by tracking which videos have been
    processed and their status.
    """

    brand_name: str
    videos: dict[str, VideoProgress] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    provider: str = ""
    total_cost_usd: float = 0.0

    def add_video(self, video_path: str | Path) -> VideoProgress:
        """Add a video to track.

        Args:
            video_path: Path to the video file

        Returns:
            VideoProgress object for the video
        """
        video_path_str = str(video_path)

        if video_path_str not in self.videos:
            self.videos[video_path_str] = VideoProgress(video_path=video_path_str)
            self.updated_at = datetime.now()

        return self.videos[video_path_str]

    def start_video(self, video_path: str | Path) -> VideoProgress:
        """Mark a video as in progress.

        Args:
            video_path: Path to the video file

        Returns:
            VideoProgress object for the video
        """
        progress = self.add_video(video_path)
        progress.status = TranscriptionStatus.IN_PROGRESS
        progress.started_at = datetime.now()
        progress.attempts += 1
        self.updated_at = datetime.now()
        return progress

    def complete_video(
        self,
        video_path: str | Path,
        transcript_path: str | Path,
        duration_seconds: float = 0.0,
        cost_usd: float | None = None,
    ) -> VideoProgress:
        """Mark a video as completed.

        Args:
            video_path: Path to the video file
            transcript_path: Path to the saved transcript
            duration_seconds: Duration of the video
            cost_usd: Cost of transcription (if API)

        Returns:
            VideoProgress object for the video
        """
        progress = self.add_video(video_path)
        progress.status = TranscriptionStatus.COMPLETED
        progress.transcript_path = str(transcript_path)
        progress.completed_at = datetime.now()
        progress.duration_seconds = duration_seconds
        progress.cost_usd = cost_usd
        progress.error_message = None

        if cost_usd:
            self.total_cost_usd += cost_usd

        self.updated_at = datetime.now()
        return progress

    def fail_video(
        self,
        video_path: str | Path,
        error_message: str,
    ) -> VideoProgress:
        """Mark a video as failed.

        Args:
            video_path: Path to the video file
            error_message: Description of the error

        Returns:
            VideoProgress object for the video
        """
        progress = self.add_video(video_path)
        progress.status = TranscriptionStatus.FAILED
        progress.completed_at = datetime.now()
        progress.error_message = error_message
        self.updated_at = datetime.now()
        return progress

    def skip_video(
        self,
        video_path: str | Path,
        reason: str = "Already transcribed",
    ) -> VideoProgress:
        """Mark a video as skipped.

        Args:
            video_path: Path to the video file
            reason: Reason for skipping

        Returns:
            VideoProgress object for the video
        """
        progress = self.add_video(video_path)
        progress.status = TranscriptionStatus.SKIPPED
        progress.error_message = reason
        self.updated_at = datetime.now()
        return progress

    def is_completed(self, video_path: str | Path) -> bool:
        """Check if a video has been successfully transcribed.

        Args:
            video_path: Path to the video file

        Returns:
            True if video has been transcribed
        """
        video_path_str = str(video_path)
        if video_path_str not in self.videos:
            return False
        return self.videos[video_path_str].status == TranscriptionStatus.COMPLETED

    def needs_processing(self, video_path: str | Path) -> bool:
        """Check if a video needs processing.

        Args:
            video_path: Path to the video file

        Returns:
            True if video needs to be transcribed
        """
        video_path_str = str(video_path)
        if video_path_str not in self.videos:
            return True

        status = self.videos[video_path_str].status
        return status in (TranscriptionStatus.PENDING, TranscriptionStatus.FAILED)

    def get_pending_videos(self) -> list[str]:
        """Get list of videos that need processing.

        Returns:
            List of video paths that need transcription
        """
        return [
            path for path, progress in self.videos.items()
            if progress.status in (TranscriptionStatus.PENDING, TranscriptionStatus.FAILED)
        ]

    def get_completed_videos(self) -> list[str]:
        """Get list of successfully transcribed videos.

        Returns:
            List of video paths that have been transcribed
        """
        return [
            path for path, progress in self.videos.items()
            if progress.status == TranscriptionStatus.COMPLETED
        ]

    def get_failed_videos(self) -> list[str]:
        """Get list of videos that failed transcription.

        Returns:
            List of video paths that failed
        """
        return [
            path for path, progress in self.videos.items()
            if progress.status == TranscriptionStatus.FAILED
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with progress statistics
        """
        total = len(self.videos)
        completed = sum(1 for v in self.videos.values() if v.status == TranscriptionStatus.COMPLETED)
        failed = sum(1 for v in self.videos.values() if v.status == TranscriptionStatus.FAILED)
        pending = sum(1 for v in self.videos.values() if v.status == TranscriptionStatus.PENDING)
        in_progress = sum(1 for v in self.videos.values() if v.status == TranscriptionStatus.IN_PROGRESS)
        skipped = sum(1 for v in self.videos.values() if v.status == TranscriptionStatus.SKIPPED)

        total_duration = sum(v.duration_seconds for v in self.videos.values() if v.status == TranscriptionStatus.COMPLETED)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "skipped": skipped,
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60.0,
            "total_cost_usd": self.total_cost_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "brand_name": self.brand_name,
            "provider": self.provider,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_cost_usd": self.total_cost_usd,
            "summary": self.get_summary(),
            "videos": {path: progress.to_dict() for path, progress in self.videos.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionProgress":
        """Create from dictionary."""
        progress = cls(
            brand_name=data["brand_name"],
            provider=data.get("provider", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            total_cost_usd=data.get("total_cost_usd", 0.0),
        )

        for path, video_data in data.get("videos", {}).items():
            progress.videos[path] = VideoProgress.from_dict(video_data)

        return progress

    def save(self, path: Path | str) -> None:
        """Save progress to JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        temp_path.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> "TranscriptionProgress":
        """Load progress from JSON file.

        Args:
            path: Path to load from

        Returns:
            TranscriptionProgress instance
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, path: Path | str, brand_name: str) -> "TranscriptionProgress":
        """Load progress from file or create new if not exists.

        Args:
            path: Path to progress file
            brand_name: Name of the brand

        Returns:
            TranscriptionProgress instance
        """
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(brand_name=brand_name)
