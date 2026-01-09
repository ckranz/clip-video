"""Data models for clip-video.

This module provides Pydantic models for brands, projects, transcripts, and clips.
"""

from __future__ import annotations

from clip_video.models.brand import Brand, CaptionStyle, CropRegion
from clip_video.models.clip import Clip, ClipStatus
from clip_video.models.project import Project, ProjectStatus, ProjectType
from clip_video.models.transcript import Transcript, TranscriptSegment, TranscriptWord

__all__ = [
    # Brand models
    "Brand",
    "CaptionStyle",
    "CropRegion",
    # Project models
    "Project",
    "ProjectStatus",
    "ProjectType",
    # Transcript models
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    # Clip models
    "Clip",
    "ClipStatus",
]
