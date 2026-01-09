"""Selection tracking for clip extraction projects.

Tracks which clips have been selected to avoid reusing the same source
video segment in a single project. Enables diversity in lyric match
mashups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from clip_video.storage import atomic_write_json, read_json
from clip_video.search import SearchResult


@dataclass
class Selection:
    """A selected clip for extraction.

    Attributes:
        target_text: The word/phrase this selection is for
        result: The search result that was selected
        line_number: Lyric line number this is for
        selected_at: Timestamp when selection was made
        extracted: Whether clip has been extracted
        output_path: Path to extracted clip if available
        notes: Optional notes about the selection
    """

    target_text: str
    result: SearchResult
    line_number: int
    selected_at: str = ""
    extracted: bool = False
    output_path: Path | None = None
    notes: str = ""

    @property
    def source_key(self) -> str:
        """Unique key identifying the source video and time range."""
        return f"{self.result.source_key}:{self.result.start:.2f}-{self.result.end:.2f}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "target_text": self.target_text,
            "result": self.result.to_dict(),
            "line_number": self.line_number,
            "selected_at": self.selected_at,
            "extracted": self.extracted,
            "output_path": str(self.output_path) if self.output_path else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Selection":
        """Create from dictionary."""
        return cls(
            target_text=data["target_text"],
            result=SearchResult.from_dict(data["result"]),
            line_number=data["line_number"],
            selected_at=data.get("selected_at", ""),
            extracted=data.get("extracted", False),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            notes=data.get("notes", ""),
        )


@dataclass
class SelectionTracker:
    """Tracks clip selections for a project to prevent reuse.

    Maintains a registry of which source video segments have been
    selected, enabling diversity in clip selection.

    Attributes:
        project_name: Name of the project
        brand_name: Brand being used
        selections: List of selections made
        excluded_videos: Set of video IDs to exclude from selection
        excluded_ranges: Dict mapping video_id to list of (start, end) tuples
    """

    project_name: str
    brand_name: str
    selections: list[Selection] = field(default_factory=list)
    excluded_videos: set[str] = field(default_factory=set)
    excluded_ranges: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # Minimum time gap between selected clips from same video (seconds)
    MIN_GAP_SECONDS: float = 5.0

    def is_available(
        self,
        result: SearchResult,
        allow_same_video: bool = True,
        min_gap: float | None = None,
    ) -> bool:
        """Check if a search result is available for selection.

        Args:
            result: SearchResult to check
            allow_same_video: If False, reject any result from already-used videos
            min_gap: Minimum time gap from existing selections (default: MIN_GAP_SECONDS)

        Returns:
            True if the result can be selected
        """
        min_gap = min_gap if min_gap is not None else self.MIN_GAP_SECONDS

        # Check if video is excluded entirely
        if result.video_id in self.excluded_videos:
            return False

        # Check if video is already used and we don't allow same video
        if not allow_same_video:
            for sel in self.selections:
                if sel.result.video_id == result.video_id:
                    return False

        # Check for overlapping or too-close ranges
        video_ranges = self.excluded_ranges.get(result.video_id, [])
        for start, end in video_ranges:
            # Check overlap
            if result.start < end and result.end > start:
                return False
            # Check too close
            if abs(result.start - end) < min_gap or abs(result.end - start) < min_gap:
                return False

        # Check against existing selections
        for sel in self.selections:
            if sel.result.video_id != result.video_id:
                continue

            # Check overlap
            if result.start < sel.result.end and result.end > sel.result.start:
                return False

            # Check too close
            if (
                abs(result.start - sel.result.end) < min_gap
                or abs(result.end - sel.result.start) < min_gap
            ):
                return False

        return True

    def select(
        self,
        target_text: str,
        result: SearchResult,
        line_number: int,
        notes: str = "",
    ) -> Selection:
        """Select a search result for extraction.

        Args:
            target_text: The word/phrase being matched
            result: SearchResult to select
            line_number: Lyric line number
            notes: Optional notes

        Returns:
            Selection object

        Raises:
            ValueError: If result is not available
        """
        from datetime import datetime

        if not self.is_available(result):
            raise ValueError(
                f"Result at {result.start}s in {result.video_id} is not available"
            )

        selection = Selection(
            target_text=target_text,
            result=result,
            line_number=line_number,
            selected_at=datetime.now().isoformat(),
            notes=notes,
        )

        self.selections.append(selection)
        return selection

    def auto_select(
        self,
        target_text: str,
        results: list[SearchResult],
        line_number: int,
        prefer_diversity: bool = True,
    ) -> Selection | None:
        """Automatically select the best available result.

        Args:
            target_text: The word/phrase being matched
            results: List of search results to choose from (sorted by rank)
            line_number: Lyric line number
            prefer_diversity: If True, prefer results from unused videos

        Returns:
            Selection if successful, None if no available results
        """
        if not results:
            return None

        # If preferring diversity, first try to find result from unused video
        if prefer_diversity:
            used_videos = {sel.result.video_id for sel in self.selections}
            for result in results:
                if result.video_id not in used_videos:
                    if self.is_available(result):
                        return self.select(target_text, result, line_number)

        # Fall back to any available result
        for result in results:
            if self.is_available(result):
                return self.select(target_text, result, line_number)

        return None

    def unselect(self, selection: Selection) -> bool:
        """Remove a selection.

        Args:
            selection: Selection to remove

        Returns:
            True if removed, False if not found
        """
        try:
            self.selections.remove(selection)
            return True
        except ValueError:
            return False

    def exclude_video(self, video_id: str) -> None:
        """Exclude an entire video from selection.

        Args:
            video_id: Video to exclude
        """
        self.excluded_videos.add(video_id)

    def exclude_range(self, video_id: str, start: float, end: float) -> None:
        """Exclude a time range from a video.

        Args:
            video_id: Video containing the range
            start: Start time in seconds
            end: End time in seconds
        """
        if video_id not in self.excluded_ranges:
            self.excluded_ranges[video_id] = []
        self.excluded_ranges[video_id].append((start, end))

    def get_selections_for_line(self, line_number: int) -> list[Selection]:
        """Get all selections for a specific line.

        Args:
            line_number: Line number to get selections for

        Returns:
            List of selections for that line
        """
        return [s for s in self.selections if s.line_number == line_number]

    def get_selections_for_target(self, target_text: str) -> list[Selection]:
        """Get all selections for a specific target.

        Args:
            target_text: Target text to get selections for

        Returns:
            List of selections for that target
        """
        return [s for s in self.selections if s.target_text == target_text]

    @property
    def used_videos(self) -> set[str]:
        """Get set of video IDs used in selections."""
        return {sel.result.video_id for sel in self.selections}

    @property
    def selection_count(self) -> int:
        """Get total number of selections."""
        return len(self.selections)

    @property
    def extracted_count(self) -> int:
        """Get number of selections that have been extracted."""
        return sum(1 for s in self.selections if s.extracted)

    def get_statistics(self) -> dict:
        """Get selection statistics.

        Returns:
            Dict with statistics
        """
        return {
            "project_name": self.project_name,
            "brand_name": self.brand_name,
            "total_selections": self.selection_count,
            "extracted_count": self.extracted_count,
            "unique_videos_used": len(self.used_videos),
            "excluded_videos": len(self.excluded_videos),
            "lines_with_selections": len(set(s.line_number for s in self.selections)),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "project_name": self.project_name,
            "brand_name": self.brand_name,
            "selections": [s.to_dict() for s in self.selections],
            "excluded_videos": list(self.excluded_videos),
            "excluded_ranges": {
                vid: [(s, e) for s, e in ranges]
                for vid, ranges in self.excluded_ranges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionTracker":
        """Create from dictionary."""
        tracker = cls(
            project_name=data["project_name"],
            brand_name=data["brand_name"],
        )

        tracker.selections = [
            Selection.from_dict(s) for s in data.get("selections", [])
        ]
        tracker.excluded_videos = set(data.get("excluded_videos", []))
        tracker.excluded_ranges = {
            vid: [(s, e) for s, e in ranges]
            for vid, ranges in data.get("excluded_ranges", {}).items()
        }

        return tracker

    def save(self, path: Path) -> None:
        """Save tracker state to file.

        Args:
            path: Path to save to
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "SelectionTracker":
        """Load tracker state from file.

        Args:
            path: Path to load from

        Returns:
            SelectionTracker instance
        """
        data = read_json(path)
        return cls.from_dict(data)


class SelectionManager:
    """Manager for selection trackers across projects.

    Handles loading, saving, and creating selection trackers for
    lyric match projects.
    """

    def __init__(self, brands_root: Path | None = None):
        """Initialize the manager.

        Args:
            brands_root: Root directory for brands
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _tracker_path(self, brand_name: str, project_name: str) -> Path:
        """Get path to tracker file."""
        return (
            self.brands_root
            / brand_name
            / "projects"
            / project_name
            / "selections.json"
        )

    def exists(self, brand_name: str, project_name: str) -> bool:
        """Check if a tracker exists.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            True if tracker exists
        """
        return self._tracker_path(brand_name, project_name).exists()

    def get(self, brand_name: str, project_name: str) -> SelectionTracker:
        """Get or create a selection tracker.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            SelectionTracker instance
        """
        path = self._tracker_path(brand_name, project_name)
        if path.exists():
            return SelectionTracker.load(path)

        return SelectionTracker(
            project_name=project_name,
            brand_name=brand_name,
        )

    def save(self, tracker: SelectionTracker) -> Path:
        """Save a selection tracker.

        Args:
            tracker: Tracker to save

        Returns:
            Path to saved file
        """
        path = self._tracker_path(tracker.brand_name, tracker.project_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tracker.save(path)
        return path

    def delete(self, brand_name: str, project_name: str) -> bool:
        """Delete a selection tracker.

        Args:
            brand_name: Brand name
            project_name: Project name

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._tracker_path(brand_name, project_name)
        if not path.exists():
            return False
        path.unlink()
        return True
