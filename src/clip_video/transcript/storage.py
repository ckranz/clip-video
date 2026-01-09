"""Enhanced transcript storage with JSON format and word-level timestamps.

Provides TranscriptStore for storing, retrieving, and managing transcripts
with full word-level timing information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from clip_video.models.transcript import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from clip_video.storage import (
    NotFoundError,
    StorageError,
    atomic_write_json,
    load_model,
    save_model,
)


class TranscriptStore:
    """Enhanced transcript storage with JSON format.

    Stores transcripts with word-level timestamps in a structured JSON format.
    Supports listing all transcripts for a brand (across all projects) for
    cross-project searching.
    """

    def __init__(self, brands_root: Path | None = None):
        """Initialize the transcript store.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _transcripts_dir(self, brand_name: str, project_name: str) -> Path:
        """Get the transcripts directory for a project."""
        return self.brands_root / brand_name / "projects" / project_name / "transcripts"

    def _transcript_path(
        self, brand_name: str, project_name: str, video_id: str
    ) -> Path:
        """Get the path for a transcript file.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier (typically filename stem)

        Returns:
            Path to the transcript JSON file
        """
        return self._transcripts_dir(brand_name, project_name) / f"{video_id}.json"

    def save(
        self,
        brand_name: str,
        project_name: str,
        video_id: str,
        transcript: Transcript,
    ) -> Path:
        """Save a transcript for a video.

        The transcript is stored as JSON with full word-level timing information.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier (typically filename stem)
            transcript: Transcript to save

        Returns:
            Path to the saved transcript file
        """
        transcripts_dir = self._transcripts_dir(brand_name, project_name)
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        path = self._transcript_path(brand_name, project_name, video_id)
        save_model(path, transcript)
        return path

    def get(
        self, brand_name: str, project_name: str, video_id: str
    ) -> Transcript:
        """Get a transcript for a video.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier (typically filename stem)

        Returns:
            The transcript

        Raises:
            NotFoundError: If transcript doesn't exist
        """
        path = self._transcript_path(brand_name, project_name, video_id)
        if not path.exists():
            raise NotFoundError(f"Transcript not found for: {video_id}")

        return load_model(path, Transcript)

    def exists(self, brand_name: str, project_name: str, video_id: str) -> bool:
        """Check if a transcript exists for a video.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier (typically filename stem)

        Returns:
            True if transcript exists
        """
        return self._transcript_path(brand_name, project_name, video_id).exists()

    def delete(
        self, brand_name: str, project_name: str, video_id: str
    ) -> bool:
        """Delete a transcript.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier (typically filename stem)

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._transcript_path(brand_name, project_name, video_id)
        if not path.exists():
            return False

        path.unlink()
        return True

    def list_for_project(self, brand_name: str, project_name: str) -> list[str]:
        """List all video IDs that have transcripts in a project.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            List of video IDs that have transcripts
        """
        transcripts_dir = self._transcripts_dir(brand_name, project_name)
        if not transcripts_dir.exists():
            return []

        return sorted([f.stem for f in transcripts_dir.glob("*.json")])

    def list_projects(self, brand_name: str) -> list[str]:
        """List all projects for a brand that have transcripts.

        Args:
            brand_name: Brand name

        Returns:
            List of project names
        """
        projects_dir = self.brands_root / brand_name / "projects"
        if not projects_dir.exists():
            return []

        projects = []
        for project_path in projects_dir.iterdir():
            if project_path.is_dir():
                transcripts_dir = project_path / "transcripts"
                if transcripts_dir.exists() and any(transcripts_dir.glob("*.json")):
                    projects.append(project_path.name)

        return sorted(projects)

    def iter_all_for_brand(
        self, brand_name: str
    ) -> Iterator[tuple[str, str, Transcript]]:
        """Iterate over all transcripts for a brand.

        Yields transcripts from all projects within the brand.

        Args:
            brand_name: Brand name

        Yields:
            Tuples of (project_name, video_id, transcript)
        """
        for project_name in self.list_projects(brand_name):
            for video_id in self.list_for_project(brand_name, project_name):
                try:
                    transcript = self.get(brand_name, project_name, video_id)
                    yield project_name, video_id, transcript
                except (NotFoundError, StorageError):
                    # Skip corrupted or inaccessible transcripts
                    continue

    def get_all_for_brand(
        self, brand_name: str
    ) -> list[tuple[str, str, Transcript]]:
        """Get all transcripts for a brand.

        Args:
            brand_name: Brand name

        Returns:
            List of tuples (project_name, video_id, transcript)
        """
        return list(self.iter_all_for_brand(brand_name))

    def count_for_brand(self, brand_name: str) -> int:
        """Count total number of transcripts for a brand.

        Args:
            brand_name: Brand name

        Returns:
            Number of transcripts
        """
        count = 0
        for project_name in self.list_projects(brand_name):
            count += len(self.list_for_project(brand_name, project_name))
        return count

    def get_transcript_metadata(
        self, brand_name: str, project_name: str, video_id: str
    ) -> dict:
        """Get transcript metadata without loading full transcript.

        Useful for quickly scanning transcripts without loading all word data.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_id: Video identifier

        Returns:
            Dict with metadata fields (video_path, language, duration, etc.)

        Raises:
            NotFoundError: If transcript doesn't exist
        """
        path = self._transcript_path(brand_name, project_name, video_id)
        if not path.exists():
            raise NotFoundError(f"Transcript not found for: {video_id}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Return only metadata fields
        return {
            "video_path": data.get("video_path", ""),
            "language": data.get("language", "en"),
            "created_at": data.get("created_at"),
            "provider": data.get("provider", ""),
            "model": data.get("model", ""),
            "duration": data.get("duration", 0.0),
            "segment_count": len(data.get("segments", [])),
        }
