"""General progress tracking with progress bar and ETA.

Provides a rich-based progress tracker for batch operations with support
for nested progress (e.g., processing videos, then clips within each video).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Generator, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")


@dataclass
class ProgressStats:
    """Statistics for a progress tracker."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def pending(self) -> int:
        """Number of pending items."""
        return self.total - self.completed - self.failed - self.skipped

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        return (self.completed / processed) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def avg_time_per_item(self) -> float:
        """Average time per completed item in seconds."""
        if self.completed == 0:
            return 0.0
        return self.elapsed_seconds / self.completed

    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimated remaining time in seconds."""
        if self.avg_time_per_item == 0:
            return 0.0
        return self.pending * self.avg_time_per_item

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": self.elapsed_seconds,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressStats":
        """Create from dictionary."""
        return cls(
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
            start_time=(
                datetime.fromisoformat(data["start_time"])
                if data.get("start_time")
                else None
            ),
            end_time=(
                datetime.fromisoformat(data["end_time"])
                if data.get("end_time")
                else None
            ),
        )


class ProgressTracker:
    """Rich-based progress tracker with ETA support.

    Supports nested progress bars for hierarchical operations like
    processing multiple videos, each with multiple clips.

    Example:
        with ProgressTracker("Processing videos") as tracker:
            for video in videos:
                with tracker.task(f"Video: {video.name}", total=len(clips)) as task:
                    for clip in clips:
                        process_clip(clip)
                        task.advance()
    """

    def __init__(
        self,
        description: str = "Processing",
        console: Console | None = None,
        show_speed: bool = True,
        transient: bool = False,
    ):
        """Initialize progress tracker.

        Args:
            description: Main description for the progress
            console: Rich console to use (creates new if None)
            show_speed: Whether to show processing speed
            transient: Whether to clear progress on completion
        """
        self.description = description
        self.console = console or Console()
        self.show_speed = show_speed
        self.transient = transient

        self._progress: Progress | None = None
        self._main_task: TaskID | None = None
        self._stats = ProgressStats()
        self._nested_level = 0
        self._task_stack: list[TaskID] = []

    @property
    def stats(self) -> ProgressStats:
        """Get current progress statistics."""
        return self._stats

    def _create_progress(self) -> Progress:
        """Create the Rich Progress object with appropriate columns."""
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[cyan]ETA:[/cyan]"),
            TimeRemainingColumn(),
        ]

        return Progress(
            *columns,
            console=self.console,
            transient=self.transient,
            refresh_per_second=10,
        )

    def __enter__(self) -> "ProgressTracker":
        """Start the progress display."""
        self._progress = self._create_progress()
        self._progress.__enter__()
        self._stats.start_time = datetime.now()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the progress display."""
        self._stats.end_time = datetime.now()
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
            self._progress = None

    def add_main_task(self, total: int, description: str | None = None) -> TaskID:
        """Add the main progress task.

        Args:
            total: Total number of items to process
            description: Override description (uses self.description if None)

        Returns:
            Task ID for the main task
        """
        if self._progress is None:
            raise RuntimeError("ProgressTracker must be used as context manager")

        self._stats.total = total
        self._main_task = self._progress.add_task(
            description or self.description,
            total=total,
        )
        return self._main_task

    def add_subtask(
        self,
        description: str,
        total: int | None = None,
    ) -> TaskID:
        """Add a subtask for nested progress.

        Args:
            description: Description for the subtask
            total: Total items in subtask (None for indeterminate)

        Returns:
            Task ID for the subtask
        """
        if self._progress is None:
            raise RuntimeError("ProgressTracker must be used as context manager")

        task_id = self._progress.add_task(
            f"  {description}",
            total=total,
        )
        self._task_stack.append(task_id)
        return task_id

    def advance(
        self,
        task_id: TaskID | None = None,
        advance: int = 1,
        description: str | None = None,
    ) -> None:
        """Advance progress on a task.

        Args:
            task_id: Task to advance (uses main task if None)
            advance: Number of items to advance
            description: Update task description
        """
        if self._progress is None:
            return

        target_task = task_id if task_id is not None else self._main_task
        if target_task is None:
            return

        update_kwargs: dict[str, Any] = {"advance": advance}
        if description:
            update_kwargs["description"] = description

        self._progress.update(target_task, **update_kwargs)

    def complete_item(self) -> None:
        """Mark one item as completed and advance main task."""
        self._stats.completed += 1
        self.advance()

    def fail_item(self) -> None:
        """Mark one item as failed and advance main task."""
        self._stats.failed += 1
        self.advance()

    def skip_item(self) -> None:
        """Mark one item as skipped and advance main task."""
        self._stats.skipped += 1
        self.advance()

    def remove_subtask(self, task_id: TaskID) -> None:
        """Remove a completed subtask.

        Args:
            task_id: Task ID to remove
        """
        if self._progress is None:
            return

        self._progress.remove_task(task_id)
        if task_id in self._task_stack:
            self._task_stack.remove(task_id)

    def update_description(self, description: str, task_id: TaskID | None = None) -> None:
        """Update task description.

        Args:
            description: New description
            task_id: Task to update (uses main task if None)
        """
        if self._progress is None:
            return

        target_task = task_id if task_id is not None else self._main_task
        if target_task is not None:
            self._progress.update(target_task, description=description)

    @contextmanager
    def task(
        self,
        description: str,
        total: int | None = None,
    ) -> Generator["SubTaskContext", None, None]:
        """Context manager for a subtask.

        Args:
            description: Subtask description
            total: Total items in subtask

        Yields:
            SubTaskContext for controlling the subtask
        """
        task_id = self.add_subtask(description, total)
        try:
            yield SubTaskContext(self, task_id)
        finally:
            self.remove_subtask(task_id)

    def process_items(
        self,
        items: list[T],
        processor: Callable[[T], bool],
        description_fn: Callable[[T], str] | None = None,
    ) -> ProgressStats:
        """Process a list of items with progress tracking.

        Args:
            items: Items to process
            processor: Function to process each item (returns True on success)
            description_fn: Optional function to get description for each item

        Returns:
            Final progress statistics
        """
        self.add_main_task(len(items))

        for item in items:
            if description_fn:
                self.update_description(description_fn(item))

            try:
                success = processor(item)
                if success:
                    self.complete_item()
                else:
                    self.fail_item()
            except Exception:
                self.fail_item()

        return self._stats


@dataclass
class SubTaskContext:
    """Context for controlling a subtask within ProgressTracker."""

    tracker: ProgressTracker
    task_id: TaskID
    _completed: int = field(default=0, init=False)

    def advance(self, amount: int = 1, description: str | None = None) -> None:
        """Advance the subtask progress.

        Args:
            amount: Number of items to advance
            description: Update description
        """
        self._completed += amount
        self.tracker.advance(self.task_id, amount, description)

    def update(self, description: str) -> None:
        """Update subtask description.

        Args:
            description: New description
        """
        self.tracker.update_description(f"  {description}", self.task_id)


class BatchProgressTracker:
    """Progress tracker for batch operations with multiple phases.

    Useful for multi-stage pipelines like:
    1. Transcription
    2. Analysis
    3. Clip Extraction
    4. Rendering

    Example:
        tracker = BatchProgressTracker(console)
        with tracker.phase("Transcription", total_videos) as phase:
            for video in videos:
                transcribe(video)
                phase.advance()
    """

    def __init__(
        self,
        console: Console | None = None,
        show_overall: bool = True,
    ):
        """Initialize batch progress tracker.

        Args:
            console: Rich console to use
            show_overall: Whether to show overall progress across phases
        """
        self.console = console or Console()
        self.show_overall = show_overall

        self._phases: list[str] = []
        self._current_phase: int = 0
        self._phase_stats: dict[str, ProgressStats] = {}
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time across all phases."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or datetime.now()
        return (end - self._start_time).total_seconds()

    def get_phase_stats(self, phase_name: str) -> ProgressStats | None:
        """Get statistics for a specific phase."""
        return self._phase_stats.get(phase_name)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all phases."""
        total_completed = sum(s.completed for s in self._phase_stats.values())
        total_failed = sum(s.failed for s in self._phase_stats.values())
        total_skipped = sum(s.skipped for s in self._phase_stats.values())

        return {
            "phases": self._phases,
            "current_phase": self._current_phase,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "elapsed_seconds": self.elapsed_seconds,
            "phase_stats": {
                name: stats.to_dict() for name, stats in self._phase_stats.items()
            },
        }

    @contextmanager
    def phase(
        self,
        name: str,
        total: int,
        description: str | None = None,
    ) -> Generator[ProgressTracker, None, None]:
        """Context manager for a processing phase.

        Args:
            name: Phase name (for tracking)
            total: Total items in this phase
            description: Display description (uses name if None)

        Yields:
            ProgressTracker for this phase
        """
        if self._start_time is None:
            self._start_time = datetime.now()

        self._phases.append(name)
        self._current_phase = len(self._phases)

        # Show phase header
        phase_num = len(self._phases)
        header = f"[{phase_num}] {description or name}"
        self.console.print(f"\n[bold cyan]{header}[/bold cyan]")

        tracker = ProgressTracker(
            description=description or name,
            console=self.console,
            transient=False,
        )

        with tracker:
            tracker.add_main_task(total)
            yield tracker

        self._phase_stats[name] = tracker.stats

    def print_summary(self) -> None:
        """Print summary of all phases."""
        self._end_time = datetime.now()
        summary = self.get_summary()

        self.console.print("\n[bold]Batch Processing Summary[/bold]")
        self.console.print(f"Total time: {timedelta(seconds=int(summary['elapsed_seconds']))}")
        self.console.print(f"Completed: [green]{summary['total_completed']}[/green]")
        self.console.print(f"Failed: [red]{summary['total_failed']}[/red]")
        self.console.print(f"Skipped: [yellow]{summary['total_skipped']}[/yellow]")


def format_eta(seconds: float) -> str:
    """Format seconds as human-readable ETA string.

    Args:
        seconds: Number of seconds remaining

    Returns:
        Formatted string like "2h 30m" or "5m 30s"
    """
    if seconds <= 0:
        return "complete"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string.

    Args:
        seconds: Number of seconds

    Returns:
        Formatted string like "2:30:00" or "5:30"
    """
    if seconds < 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
