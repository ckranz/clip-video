"""Project model for clip-video.

A Project represents a specific video editing task within a brand.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    """Type of video project."""

    HIGHLIGHTS = "highlights"  # Extract highlight clips from source videos
    LYRIC_MATCH = "lyric_match"  # Match visuals to lyrics/audio


class ProjectStatus(str, Enum):
    """Current status of a project."""

    CREATED = "created"  # Project just created
    TRANSCRIBING = "transcribing"  # Transcription in progress
    TRANSCRIBED = "transcribed"  # Transcription complete
    ANALYZING = "analyzing"  # LLM analyzing for clips
    ANALYZED = "analyzed"  # Analysis complete, clips identified
    RENDERING = "rendering"  # Rendering clips
    COMPLETE = "complete"  # All clips rendered
    ERROR = "error"  # An error occurred


class Project(BaseModel):
    """Configuration and state for a video project.

    A project tracks the full pipeline from source videos through
    transcription, analysis, and clip rendering.
    """

    name: str
    brand_name: str
    description: str = ""
    project_type: ProjectType = ProjectType.HIGHLIGHTS
    status: ProjectStatus = ProjectStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Source videos (paths relative to project directory)
    source_videos: list[str] = Field(default_factory=list)

    # For lyric match projects
    lyrics_file: str | None = None
    audio_file: str | None = None

    # Transcription results (paths to transcript JSON files)
    transcripts: dict[str, str] = Field(default_factory=dict)  # video_path -> transcript_path

    # Generated clips (list of clip IDs)
    clip_ids: list[str] = Field(default_factory=list)

    # Custom settings that override brand defaults
    custom_crop_region: dict[str, int] | None = None
    custom_caption_style: dict[str, Any] | None = None

    # Processing state
    current_video_index: int = 0
    current_clip_index: int = 0
    error_message: str | None = None

    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now()

    def set_status(self, status: ProjectStatus, error: str | None = None) -> None:
        """Update project status.

        Args:
            status: New status to set
            error: Optional error message if status is ERROR
        """
        self.status = status
        if status == ProjectStatus.ERROR:
            self.error_message = error
        else:
            self.error_message = None
        self.update_timestamp()

    def add_source_video(self, video_path: str) -> None:
        """Add a source video to the project.

        Args:
            video_path: Path to the video file (relative to project dir)
        """
        if video_path not in self.source_videos:
            self.source_videos.append(video_path)
            self.update_timestamp()

    def remove_source_video(self, video_path: str) -> bool:
        """Remove a source video from the project.

        Args:
            video_path: Path to the video file to remove

        Returns:
            True if video was found and removed, False otherwise
        """
        if video_path in self.source_videos:
            self.source_videos.remove(video_path)
            # Also remove associated transcript if any
            if video_path in self.transcripts:
                del self.transcripts[video_path]
            self.update_timestamp()
            return True
        return False

    def set_transcript(self, video_path: str, transcript_path: str) -> None:
        """Associate a transcript with a source video.

        Args:
            video_path: Path to the source video
            transcript_path: Path to the transcript JSON file
        """
        self.transcripts[video_path] = transcript_path
        self.update_timestamp()

    def add_clip(self, clip_id: str) -> None:
        """Add a clip ID to the project.

        Args:
            clip_id: Unique identifier for the clip
        """
        if clip_id not in self.clip_ids:
            self.clip_ids.append(clip_id)
            self.update_timestamp()

    def get_project_dir(self, brands_root: Path) -> Path:
        """Get the project directory path.

        Args:
            brands_root: Root directory containing all brands

        Returns:
            Path to this project's directory
        """
        return brands_root / self.brand_name / "projects" / self.name

    def is_complete(self) -> bool:
        """Check if the project has completed processing."""
        return self.status == ProjectStatus.COMPLETE

    def has_error(self) -> bool:
        """Check if the project has encountered an error."""
        return self.status == ProjectStatus.ERROR
