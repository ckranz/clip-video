"""Storage layer for clip-video.

Provides atomic file operations and manager classes for CRUD operations
on brands, projects, transcripts, and clips.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from clip_video.models.brand import Brand
from clip_video.models.clip import Clip
from clip_video.models.project import Project
from clip_video.models.transcript import Transcript

T = TypeVar("T", bound=BaseModel)


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class NotFoundError(StorageError):
    """Raised when a requested resource is not found."""

    pass


class AlreadyExistsError(StorageError):
    """Raised when trying to create a resource that already exists."""

    pass


def atomic_write(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Write data to a file atomically.

    Writes to a temporary file in the same directory, then renames to target.
    This prevents data corruption from interrupted writes (e.g., kill -9).

    Args:
        path: Target file path
        data: String data to write
        encoding: File encoding (default utf-8)

    Raises:
        StorageError: If the write operation fails
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    # Using same directory ensures same filesystem for rename
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=path.stem + "_",
            dir=path.parent,
        )
        # Write data
        os.write(fd, data.encode(encoding))
        os.fsync(fd)  # Ensure data is flushed to disk
        os.close(fd)
        fd = None

        # Atomic rename (works on both Windows and Unix)
        # On Windows, we need to remove target first if it exists
        if os.name == "nt" and path.exists():
            path.unlink()
        os.rename(temp_path, path)
        temp_path = None

    except Exception as e:
        raise StorageError(f"Failed to write {path}: {e}") from e
    finally:
        # Clean up on failure
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def atomic_write_json(path: Path, data: dict | list, indent: int = 2) -> None:
    """Write JSON data to a file atomically.

    Args:
        path: Target file path
        data: Data to serialize as JSON
        indent: JSON indentation level (default 2)
    """
    json_str = json.dumps(data, indent=indent, default=str)
    atomic_write(path, json_str)


def read_json(path: Path) -> dict | list:
    """Read JSON data from a file.

    Args:
        path: File path to read

    Returns:
        Parsed JSON data

    Raises:
        NotFoundError: If file doesn't exist
        StorageError: If file is invalid JSON
    """
    if not path.exists():
        raise NotFoundError(f"File not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise StorageError(f"Invalid JSON in {path}: {e}") from e


def save_model(path: Path, model: BaseModel) -> None:
    """Save a Pydantic model to JSON file atomically.

    Args:
        path: Target file path
        model: Pydantic model to save
    """
    atomic_write_json(path, model.model_dump(mode="json"))


def load_model(path: Path, model_class: type[T]) -> T:
    """Load a Pydantic model from JSON file.

    Args:
        path: File path to read
        model_class: Pydantic model class to instantiate

    Returns:
        Instance of the model class

    Raises:
        NotFoundError: If file doesn't exist
        StorageError: If file is invalid or doesn't match model
    """
    data = read_json(path)
    try:
        return model_class.model_validate(data)
    except Exception as e:
        raise StorageError(f"Invalid data in {path}: {e}") from e


class BrandManager:
    """Manager for Brand CRUD operations."""

    def __init__(self, brands_root: Path | None = None):
        """Initialize the brand manager.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _brand_dir(self, name: str) -> Path:
        """Get the directory path for a brand."""
        return self.brands_root / name

    def _config_path(self, name: str) -> Path:
        """Get the config file path for a brand."""
        return self._brand_dir(name) / "config.json"

    def exists(self, name: str) -> bool:
        """Check if a brand exists.

        Args:
            name: Brand name

        Returns:
            True if brand exists
        """
        return self._config_path(name).exists()

    def create(self, brand: Brand) -> Brand:
        """Create a new brand.

        Args:
            brand: Brand to create

        Returns:
            The created brand

        Raises:
            AlreadyExistsError: If brand already exists
        """
        if self.exists(brand.name):
            raise AlreadyExistsError(f"Brand already exists: {brand.name}")

        brand_dir = self._brand_dir(brand.name)
        brand_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (brand_dir / "projects").mkdir(exist_ok=True)
        (brand_dir / "assets").mkdir(exist_ok=True)

        save_model(self._config_path(brand.name), brand)
        return brand

    def get(self, name: str) -> Brand:
        """Get a brand by name.

        Args:
            name: Brand name

        Returns:
            The brand

        Raises:
            NotFoundError: If brand doesn't exist
        """
        if not self.exists(name):
            raise NotFoundError(f"Brand not found: {name}")

        return load_model(self._config_path(name), Brand)

    def save(self, brand: Brand) -> Brand:
        """Save/update a brand.

        Args:
            brand: Brand to save

        Returns:
            The saved brand
        """
        brand.update_timestamp()
        save_model(self._config_path(brand.name), brand)
        return brand

    def delete(self, name: str) -> bool:
        """Delete a brand and all its contents.

        Args:
            name: Brand name

        Returns:
            True if deleted, False if didn't exist
        """
        brand_dir = self._brand_dir(name)
        if not brand_dir.exists():
            return False

        import shutil

        shutil.rmtree(brand_dir)
        return True

    def list(self) -> list[str]:
        """List all brand names.

        Returns:
            Sorted list of brand names
        """
        if not self.brands_root.exists():
            return []

        brands = []
        for path in self.brands_root.iterdir():
            if path.is_dir() and self._config_path(path.name).exists():
                brands.append(path.name)

        return sorted(brands)

    def list_all(self) -> list[Brand]:
        """List all brands with full details.

        Returns:
            List of Brand objects
        """
        return [self.get(name) for name in self.list()]


class ProjectManager:
    """Manager for Project CRUD operations."""

    def __init__(self, brands_root: Path | None = None):
        """Initialize the project manager.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _project_dir(self, brand_name: str, project_name: str) -> Path:
        """Get the directory path for a project."""
        return self.brands_root / brand_name / "projects" / project_name

    def _config_path(self, brand_name: str, project_name: str) -> Path:
        """Get the config file path for a project."""
        return self._project_dir(brand_name, project_name) / "project.json"

    def exists(self, brand_name: str, project_name: str) -> bool:
        """Check if a project exists.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            True if project exists
        """
        return self._config_path(brand_name, project_name).exists()

    def create(self, project: Project) -> Project:
        """Create a new project.

        Args:
            project: Project to create

        Returns:
            The created project

        Raises:
            AlreadyExistsError: If project already exists
            NotFoundError: If brand doesn't exist
        """
        # Check brand exists
        brand_config = self.brands_root / project.brand_name / "config.json"
        if not brand_config.exists():
            raise NotFoundError(f"Brand not found: {project.brand_name}")

        if self.exists(project.brand_name, project.name):
            raise AlreadyExistsError(
                f"Project already exists: {project.brand_name}/{project.name}"
            )

        project_dir = self._project_dir(project.brand_name, project.name)
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (project_dir / "videos").mkdir(exist_ok=True)
        (project_dir / "transcripts").mkdir(exist_ok=True)
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "output").mkdir(exist_ok=True)

        save_model(self._config_path(project.brand_name, project.name), project)
        return project

    def get(self, brand_name: str, project_name: str) -> Project:
        """Get a project by name.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            The project

        Raises:
            NotFoundError: If project doesn't exist
        """
        if not self.exists(brand_name, project_name):
            raise NotFoundError(f"Project not found: {brand_name}/{project_name}")

        return load_model(self._config_path(brand_name, project_name), Project)

    def save(self, project: Project) -> Project:
        """Save/update a project.

        Args:
            project: Project to save

        Returns:
            The saved project
        """
        project.update_timestamp()
        save_model(self._config_path(project.brand_name, project.name), project)
        return project

    def delete(self, brand_name: str, project_name: str) -> bool:
        """Delete a project and all its contents.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            True if deleted, False if didn't exist
        """
        project_dir = self._project_dir(brand_name, project_name)
        if not project_dir.exists():
            return False

        import shutil

        shutil.rmtree(project_dir)
        return True

    def list(self, brand_name: str) -> list[str]:
        """List all project names for a brand.

        Args:
            brand_name: Brand name

        Returns:
            Sorted list of project names
        """
        projects_dir = self.brands_root / brand_name / "projects"
        if not projects_dir.exists():
            return []

        projects = []
        for path in projects_dir.iterdir():
            if path.is_dir() and self._config_path(brand_name, path.name).exists():
                projects.append(path.name)

        return sorted(projects)

    def list_all(self, brand_name: str) -> list[Project]:
        """List all projects for a brand with full details.

        Args:
            brand_name: Brand name

        Returns:
            List of Project objects
        """
        return [self.get(brand_name, name) for name in self.list(brand_name)]


class TranscriptManager:
    """Manager for Transcript CRUD operations."""

    def __init__(self, brands_root: Path | None = None):
        """Initialize the transcript manager.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _transcripts_dir(self, brand_name: str, project_name: str) -> Path:
        """Get the transcripts directory for a project."""
        return self.brands_root / brand_name / "projects" / project_name / "transcripts"

    def _transcript_path(
        self, brand_name: str, project_name: str, video_filename: str
    ) -> Path:
        """Get the path for a transcript file."""
        # Use video filename (without extension) as transcript name
        base_name = Path(video_filename).stem
        return self._transcripts_dir(brand_name, project_name) / f"{base_name}.json"

    def save(
        self,
        brand_name: str,
        project_name: str,
        video_filename: str,
        transcript: Transcript,
    ) -> Path:
        """Save a transcript for a video.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_filename: Name of the source video file
            transcript: Transcript to save

        Returns:
            Path to the saved transcript file
        """
        transcripts_dir = self._transcripts_dir(brand_name, project_name)
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        path = self._transcript_path(brand_name, project_name, video_filename)
        save_model(path, transcript)
        return path

    def get(
        self, brand_name: str, project_name: str, video_filename: str
    ) -> Transcript:
        """Get a transcript for a video.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_filename: Name of the source video file

        Returns:
            The transcript

        Raises:
            NotFoundError: If transcript doesn't exist
        """
        path = self._transcript_path(brand_name, project_name, video_filename)
        if not path.exists():
            raise NotFoundError(f"Transcript not found for: {video_filename}")

        return load_model(path, Transcript)

    def exists(self, brand_name: str, project_name: str, video_filename: str) -> bool:
        """Check if a transcript exists for a video.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_filename: Name of the source video file

        Returns:
            True if transcript exists
        """
        return self._transcript_path(brand_name, project_name, video_filename).exists()

    def delete(
        self, brand_name: str, project_name: str, video_filename: str
    ) -> bool:
        """Delete a transcript.

        Args:
            brand_name: Brand name
            project_name: Project name
            video_filename: Name of the source video file

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._transcript_path(brand_name, project_name, video_filename)
        if not path.exists():
            return False

        path.unlink()
        return True

    def list(self, brand_name: str, project_name: str) -> list[str]:
        """List all video filenames that have transcripts.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            List of video filenames (stems) that have transcripts
        """
        transcripts_dir = self._transcripts_dir(brand_name, project_name)
        if not transcripts_dir.exists():
            return []

        return sorted([f.stem for f in transcripts_dir.glob("*.json")])


class ClipManager:
    """Manager for Clip CRUD operations."""

    def __init__(self, brands_root: Path | None = None):
        """Initialize the clip manager.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _clips_dir(self, brand_name: str, project_name: str) -> Path:
        """Get the clips directory for a project."""
        return self.brands_root / brand_name / "projects" / project_name / "clips"

    def _clip_path(self, brand_name: str, project_name: str, clip_id: str) -> Path:
        """Get the path for a clip metadata file."""
        return self._clips_dir(brand_name, project_name) / f"{clip_id}.json"

    def save(self, clip: Clip) -> Path:
        """Save a clip.

        Args:
            clip: Clip to save

        Returns:
            Path to the saved clip file
        """
        clips_dir = self._clips_dir(clip.brand_name, clip.project_name)
        clips_dir.mkdir(parents=True, exist_ok=True)

        clip.update_timestamp()
        path = self._clip_path(clip.brand_name, clip.project_name, clip.id)
        save_model(path, clip)
        return path

    def get(self, brand_name: str, project_name: str, clip_id: str) -> Clip:
        """Get a clip by ID.

        Args:
            brand_name: Brand name
            project_name: Project name
            clip_id: Clip ID

        Returns:
            The clip

        Raises:
            NotFoundError: If clip doesn't exist
        """
        path = self._clip_path(brand_name, project_name, clip_id)
        if not path.exists():
            raise NotFoundError(f"Clip not found: {clip_id}")

        return load_model(path, Clip)

    def exists(self, brand_name: str, project_name: str, clip_id: str) -> bool:
        """Check if a clip exists.

        Args:
            brand_name: Brand name
            project_name: Project name
            clip_id: Clip ID

        Returns:
            True if clip exists
        """
        return self._clip_path(brand_name, project_name, clip_id).exists()

    def delete(self, brand_name: str, project_name: str, clip_id: str) -> bool:
        """Delete a clip.

        Args:
            brand_name: Brand name
            project_name: Project name
            clip_id: Clip ID

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._clip_path(brand_name, project_name, clip_id)
        if not path.exists():
            return False

        path.unlink()
        return True

    def list(self, brand_name: str, project_name: str) -> list[str]:
        """List all clip IDs for a project.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            List of clip IDs
        """
        clips_dir = self._clips_dir(brand_name, project_name)
        if not clips_dir.exists():
            return []

        return sorted([f.stem for f in clips_dir.glob("*.json")])

    def list_all(self, brand_name: str, project_name: str) -> list[Clip]:
        """List all clips for a project with full details.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            List of Clip objects
        """
        return [
            self.get(brand_name, project_name, clip_id)
            for clip_id in self.list(brand_name, project_name)
        ]

    def list_by_status(
        self, brand_name: str, project_name: str, status: str
    ) -> list[Clip]:
        """List clips with a specific status.

        Args:
            brand_name: Brand name
            project_name: Project name
            status: Status to filter by

        Returns:
            List of Clip objects with the specified status
        """
        from clip_video.models.clip import ClipStatus

        target_status = ClipStatus(status)
        return [
            clip
            for clip in self.list_all(brand_name, project_name)
            if clip.status == target_status
        ]


# Convenience functions for getting managers with default paths


def get_brand_manager() -> BrandManager:
    """Get a BrandManager with default path."""
    return BrandManager()


def get_project_manager() -> ProjectManager:
    """Get a ProjectManager with default path."""
    return ProjectManager()


def get_transcript_manager() -> TranscriptManager:
    """Get a TranscriptManager with default path."""
    return TranscriptManager()


def get_clip_manager() -> ClipManager:
    """Get a ClipManager with default path."""
    return ClipManager()
