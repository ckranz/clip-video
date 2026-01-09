"""Clip model for clip-video.

A Clip represents a segment of video to be extracted and processed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ClipStatus(str, Enum):
    """Status of a clip in the processing pipeline."""

    PENDING = "pending"  # Identified but not yet processed
    APPROVED = "approved"  # User approved for rendering
    REJECTED = "rejected"  # User rejected
    RENDERING = "rendering"  # Currently being rendered
    COMPLETE = "complete"  # Rendering complete
    ERROR = "error"  # Error during processing


def generate_clip_id() -> str:
    """Generate a unique clip ID."""
    return str(uuid4())[:8]


class Clip(BaseModel):
    """A video clip to be extracted and processed.

    Represents a segment from a source video that has been identified
    for extraction, either by LLM analysis or manual selection.
    """

    id: str = Field(default_factory=generate_clip_id)
    project_name: str
    brand_name: str
    source_video: str  # Path to source video

    # Timing
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds

    # Metadata
    title: str = ""  # Optional title for the clip
    description: str = ""  # Why this clip was selected
    tags: list[str] = Field(default_factory=list)

    # Processing state
    status: ClipStatus = ClipStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Output
    output_path: str | None = None  # Path to rendered clip
    thumbnail_path: str | None = None  # Path to thumbnail image

    # Transcript data for this clip
    transcript_text: str = ""  # Text spoken during this clip
    transcript_words: list[dict[str, Any]] = Field(
        default_factory=list
    )  # Word timing data

    # LLM analysis data
    highlight_score: float = 0.0  # How "interesting" the clip is (0-1)
    highlight_reason: str = ""  # Why the LLM selected this clip
    suggested_caption: str = ""  # LLM-suggested caption/title

    # Processing options
    apply_captions: bool = True
    apply_crop: bool = True  # Crop to portrait
    custom_crop: dict[str, int] | None = None  # Override brand crop region

    # Error tracking
    error_message: str | None = None
    retry_count: int = 0

    @property
    def duration(self) -> float:
        """Get the duration of this clip in seconds."""
        return self.end_time - self.start_time

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now()

    def set_status(self, status: ClipStatus, error: str | None = None) -> None:
        """Update clip status.

        Args:
            status: New status to set
            error: Optional error message if status is ERROR
        """
        self.status = status
        if status == ClipStatus.ERROR:
            self.error_message = error
            self.retry_count += 1
        else:
            self.error_message = None
        self.update_timestamp()

    def approve(self) -> None:
        """Approve this clip for rendering."""
        self.set_status(ClipStatus.APPROVED)

    def reject(self) -> None:
        """Reject this clip."""
        self.set_status(ClipStatus.REJECTED)

    def mark_complete(self, output_path: str, thumbnail_path: str | None = None) -> None:
        """Mark clip as complete with output paths.

        Args:
            output_path: Path to the rendered clip file
            thumbnail_path: Optional path to thumbnail image
        """
        self.output_path = output_path
        self.thumbnail_path = thumbnail_path
        self.set_status(ClipStatus.COMPLETE)

    def is_pending(self) -> bool:
        """Check if clip is waiting for approval."""
        return self.status == ClipStatus.PENDING

    def is_ready_for_render(self) -> bool:
        """Check if clip is approved and ready for rendering."""
        return self.status == ClipStatus.APPROVED

    def is_complete(self) -> bool:
        """Check if clip has been rendered."""
        return self.status == ClipStatus.COMPLETE

    def has_error(self) -> bool:
        """Check if clip has encountered an error."""
        return self.status == ClipStatus.ERROR

    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if clip can be retried after error.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if retry is possible
        """
        return self.has_error() and self.retry_count < max_retries

    def get_output_filename(self) -> str:
        """Generate a filename for the output clip.

        Returns:
            Filename based on project, id, and title
        """
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in self.title)
        safe_title = safe_title.strip().replace(" ", "_")[:30]
        if safe_title:
            return f"{self.project_name}_{self.id}_{safe_title}.mp4"
        return f"{self.project_name}_{self.id}.mp4"
