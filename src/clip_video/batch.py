"""Batch processing orchestration for highlights mode.

Handles batch processing of multiple videos with:
- Parallel processing where possible
- Progress tracking across all videos
- Failure handling that doesn't stop the batch
- Summary report generation
- Resume capability for interrupted batches
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

from clip_video.modes.highlights import (
    HighlightsConfig,
    HighlightsProcessor,
    HighlightsProject,
    HighlightClip,
)
from clip_video.storage import atomic_write_json, read_json
from clip_video.transcription import TranscriptionSegment


class VideoStatus(str, Enum):
    """Status of a video in the batch."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VideoResult:
    """Result of processing a single video.

    Attributes:
        video_path: Path to the source video
        status: Processing status
        project_name: Name of the highlights project
        clips_generated: Number of clips generated
        error_message: Error message if failed
        started_at: When processing started
        completed_at: When processing completed
    """

    video_path: Path
    status: VideoStatus = VideoStatus.PENDING
    project_name: str = ""
    clips_generated: int = 0
    error_message: str = ""
    started_at: str = ""
    completed_at: str = ""
    clip_paths: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Calculate processing duration in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at)
        return (end - start).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "video_path": str(self.video_path),
            "status": self.status.value,
            "project_name": self.project_name,
            "clips_generated": self.clips_generated,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "clip_paths": self.clip_paths,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoResult":
        """Create from dictionary."""
        return cls(
            video_path=Path(data["video_path"]),
            status=VideoStatus(data.get("status", "pending")),
            project_name=data.get("project_name", ""),
            clips_generated=data.get("clips_generated", 0),
            error_message=data.get("error_message", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            clip_paths=data.get("clip_paths", []),
        )


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        max_parallel: Maximum parallel video processing
        continue_on_error: Continue processing if a video fails
        skip_completed: Skip videos that have already been processed
        highlights_config: Configuration for highlights processing
    """

    max_parallel: int = 4
    continue_on_error: bool = True
    skip_completed: bool = True
    highlights_config: HighlightsConfig = field(default_factory=HighlightsConfig)


@dataclass
class BatchJob:
    """A batch processing job.

    Tracks the state of a batch job including:
    - Videos to process
    - Results for each video
    - Overall progress and statistics

    Attributes:
        name: Job name
        brand_name: Brand name for all videos
        video_paths: List of video paths to process
        results: Results for each video
        config: Batch configuration
        created_at: When job was created
        updated_at: Last update time
    """

    name: str
    brand_name: str
    video_paths: list[Path]
    results: dict[str, VideoResult] = field(default_factory=dict)
    config: BatchConfig = field(default_factory=BatchConfig)
    created_at: str = ""
    updated_at: str = ""
    _job_root: Path | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

        # Initialize results for any new videos
        for video_path in self.video_paths:
            key = str(video_path)
            if key not in self.results:
                self.results[key] = VideoResult(video_path=video_path)

    @property
    def job_root(self) -> Path:
        """Get the job root directory."""
        if self._job_root:
            return self._job_root
        return Path.cwd() / "brands" / self.brand_name / "batch_jobs" / self.name

    @job_root.setter
    def job_root(self, value: Path):
        """Set the job root directory."""
        self._job_root = value

    @property
    def state_file(self) -> Path:
        """Get the state file path."""
        return self.job_root / "batch_state.json"

    @property
    def report_file(self) -> Path:
        """Get the report file path."""
        return self.job_root / "batch_report.json"

    @property
    def total_videos(self) -> int:
        """Total number of videos in the batch."""
        return len(self.video_paths)

    @property
    def completed_count(self) -> int:
        """Number of completed videos."""
        return sum(
            1 for r in self.results.values()
            if r.status == VideoStatus.COMPLETED
        )

    @property
    def failed_count(self) -> int:
        """Number of failed videos."""
        return sum(
            1 for r in self.results.values()
            if r.status == VideoStatus.FAILED
        )

    @property
    def pending_count(self) -> int:
        """Number of pending videos."""
        return sum(
            1 for r in self.results.values()
            if r.status == VideoStatus.PENDING
        )

    @property
    def progress_percent(self) -> float:
        """Overall progress percentage."""
        if self.total_videos == 0:
            return 100.0
        processed = self.completed_count + self.failed_count
        return (processed / self.total_videos) * 100

    @property
    def total_clips_generated(self) -> int:
        """Total clips generated across all videos."""
        return sum(r.clips_generated for r in self.results.values())

    def get_pending_videos(self) -> list[Path]:
        """Get list of videos still pending processing."""
        return [
            Path(key) for key, result in self.results.items()
            if result.status == VideoStatus.PENDING
        ]

    def get_failed_videos(self) -> list[tuple[Path, str]]:
        """Get list of failed videos with error messages."""
        return [
            (Path(key), result.error_message)
            for key, result in self.results.items()
            if result.status == VideoStatus.FAILED
        ]

    def get_summary(self) -> dict:
        """Get job summary statistics."""
        return {
            "job_name": self.name,
            "brand_name": self.brand_name,
            "total_videos": self.total_videos,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "pending": self.pending_count,
            "progress_percent": round(self.progress_percent, 1),
            "total_clips_generated": self.total_clips_generated,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "brand_name": self.brand_name,
            "video_paths": [str(p) for p in self.video_paths],
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict, config: BatchConfig | None = None) -> "BatchJob":
        """Create from dictionary."""
        results = {
            k: VideoResult.from_dict(v)
            for k, v in data.get("results", {}).items()
        }

        job = cls(
            name=data["name"],
            brand_name=data["brand_name"],
            video_paths=[Path(p) for p in data.get("video_paths", [])],
            results=results,
            config=config or BatchConfig(),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        return job

    def save(self) -> None:
        """Save job state to disk."""
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        atomic_write_json(self.state_file, self.to_dict())

    @classmethod
    def load(cls, state_file: Path, config: BatchConfig | None = None) -> "BatchJob":
        """Load job from state file."""
        data = read_json(state_file)
        job = cls.from_dict(data, config)
        job._job_root = state_file.parent
        return job

    def generate_report(self) -> dict:
        """Generate a comprehensive batch report."""
        report = {
            "summary": self.get_summary(),
            "videos": {
                "completed": [
                    {
                        "video_path": str(r.video_path),
                        "project_name": r.project_name,
                        "clips_generated": r.clips_generated,
                        "clip_paths": r.clip_paths,
                        "duration_seconds": r.duration,
                    }
                    for r in self.results.values()
                    if r.status == VideoStatus.COMPLETED
                ],
                "failed": [
                    {
                        "video_path": str(r.video_path),
                        "error": r.error_message,
                    }
                    for r in self.results.values()
                    if r.status == VideoStatus.FAILED
                ],
                "pending": [
                    str(r.video_path)
                    for r in self.results.values()
                    if r.status == VideoStatus.PENDING
                ],
            },
            "all_clips": [
                clip_path
                for r in self.results.values()
                if r.status == VideoStatus.COMPLETED
                for clip_path in r.clip_paths
            ],
            "generated_at": datetime.now().isoformat(),
        }

        # Save report
        atomic_write_json(self.report_file, report)

        return report


class BatchProcessor:
    """Processes batches of videos for highlights extraction.

    Orchestrates batch processing with:
    - Parallel video processing
    - Progress tracking
    - Error handling
    - Resume capability
    """

    def __init__(
        self,
        config: BatchConfig | None = None,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ):
        """Initialize the processor.

        Args:
            config: Batch configuration
            progress_callback: Optional callback for progress updates
                Signature: (stage, current, total, message)
        """
        self.config = config or BatchConfig()
        self.progress_callback = progress_callback

    def _report_progress(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, current, total, message)

    def create_job(
        self,
        name: str,
        brand_name: str,
        video_paths: list[Path],
        job_root: Path | None = None,
    ) -> BatchJob:
        """Create a new batch job.

        Args:
            name: Job name
            brand_name: Brand name
            video_paths: List of video paths
            job_root: Optional custom job root

        Returns:
            New BatchJob
        """
        job = BatchJob(
            name=name,
            brand_name=brand_name,
            video_paths=video_paths,
            config=self.config,
        )

        if job_root:
            job.job_root = job_root

        job.job_root.mkdir(parents=True, exist_ok=True)
        job.save()

        return job

    def load_video_list(self, list_file: Path) -> list[Path]:
        """Load video paths from a list file.

        Supports:
        - One path per line in a text file
        - JSON array of paths
        - Directory path (finds all videos)

        Args:
            list_file: Path to list file or directory

        Returns:
            List of video paths
        """
        if list_file.is_dir():
            # Find all video files in directory
            video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            return [
                f for f in list_file.iterdir()
                if f.suffix.lower() in video_extensions
            ]

        content = list_file.read_text(encoding="utf-8").strip()

        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [Path(p) for p in data]
        except json.JSONDecodeError:
            pass

        # Parse as line-separated paths
        paths = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))

        return paths

    def _process_single_video(
        self,
        job: BatchJob,
        video_path: Path,
        description_path: Path | None = None,
        transcript_segments: list[TranscriptionSegment] | None = None,
    ) -> VideoResult:
        """Process a single video.

        Args:
            job: Parent batch job
            video_path: Video to process
            description_path: Optional description file
            transcript_segments: Optional transcript segments

        Returns:
            VideoResult with processing outcome
        """
        key = str(video_path)
        result = job.results.get(key, VideoResult(video_path=video_path))

        # Skip if already completed and config says so
        if self.config.skip_completed and result.status == VideoStatus.COMPLETED:
            return result

        result.status = VideoStatus.IN_PROGRESS
        result.started_at = datetime.now().isoformat()

        try:
            # Create project name from video filename
            project_name = video_path.stem

            # Create highlights processor
            processor = HighlightsProcessor(config=self.config.highlights_config)

            # Create and process project
            project = processor.create_project(
                name=project_name,
                brand_name=job.brand_name,
                video_path=video_path,
                description_path=description_path,
            )

            # Run the full pipeline
            processor.process(
                project=project,
                transcript_segments=transcript_segments,
            )

            # Update result
            result.status = VideoStatus.COMPLETED
            result.project_name = project_name
            result.clips_generated = len(project.clips)
            result.clip_paths = [
                str(clip.final_clip_path)
                for clip in project.clips
                if clip.final_clip_path
            ]

        except Exception as e:
            result.status = VideoStatus.FAILED
            result.error_message = f"{type(e).__name__}: {str(e)}"

            if not self.config.continue_on_error:
                raise

        finally:
            result.completed_at = datetime.now().isoformat()
            job.results[key] = result
            job.save()

        return result

    def process_sequential(
        self,
        job: BatchJob,
        description_dir: Path | None = None,
    ) -> BatchJob:
        """Process all videos sequentially.

        Args:
            job: Batch job to process
            description_dir: Optional directory with description files

        Returns:
            Updated batch job
        """
        pending = job.get_pending_videos()
        total = len(pending)

        for i, video_path in enumerate(pending):
            self._report_progress(
                "processing",
                i + 1,
                total,
                f"Processing {video_path.name}",
            )

            # Find description file if directory provided
            description_path = None
            if description_dir:
                desc_file = description_dir / f"{video_path.stem}.txt"
                if desc_file.exists():
                    description_path = desc_file

            self._process_single_video(
                job=job,
                video_path=video_path,
                description_path=description_path,
            )

        job.generate_report()
        return job

    def process_parallel(
        self,
        job: BatchJob,
        description_dir: Path | None = None,
    ) -> BatchJob:
        """Process videos in parallel.

        Args:
            job: Batch job to process
            description_dir: Optional directory with description files

        Returns:
            Updated batch job
        """
        pending = job.get_pending_videos()
        total = len(pending)
        completed = 0

        def process_video(video_path: Path) -> VideoResult:
            description_path = None
            if description_dir:
                desc_file = description_dir / f"{video_path.stem}.txt"
                if desc_file.exists():
                    description_path = desc_file

            return self._process_single_video(
                job=job,
                video_path=video_path,
                description_path=description_path,
            )

        with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            futures = {
                executor.submit(process_video, path): path
                for path in pending
            }

            for future in as_completed(futures):
                video_path = futures[future]
                completed += 1

                try:
                    result = future.result()
                    status = "completed" if result.status == VideoStatus.COMPLETED else "failed"
                except Exception as e:
                    status = "error"

                self._report_progress(
                    "processing",
                    completed,
                    total,
                    f"{status}: {video_path.name}",
                )

        job.generate_report()
        return job

    def process(
        self,
        job: BatchJob,
        description_dir: Path | None = None,
        parallel: bool = True,
    ) -> BatchJob:
        """Process the batch job.

        Args:
            job: Batch job to process
            description_dir: Optional directory with description files
            parallel: Whether to use parallel processing

        Returns:
            Updated batch job
        """
        self._report_progress("start", 0, job.total_videos, f"Starting batch: {job.name}")

        if parallel and self.config.max_parallel > 1:
            result = self.process_parallel(job, description_dir)
        else:
            result = self.process_sequential(job, description_dir)

        self._report_progress(
            "complete",
            job.completed_count,
            job.total_videos,
            f"Batch complete: {job.completed_count}/{job.total_videos} successful",
        )

        return result

    def resume(
        self,
        job: BatchJob,
        description_dir: Path | None = None,
        parallel: bool = True,
    ) -> BatchJob:
        """Resume an interrupted batch job.

        Args:
            job: Batch job to resume
            description_dir: Optional directory with description files
            parallel: Whether to use parallel processing

        Returns:
            Updated batch job
        """
        pending_count = job.pending_count

        if pending_count == 0:
            self._report_progress(
                "complete",
                job.total_videos,
                job.total_videos,
                "All videos already processed",
            )
            return job

        self._report_progress(
            "resume",
            job.completed_count,
            job.total_videos,
            f"Resuming batch: {pending_count} videos remaining",
        )

        return self.process(job, description_dir, parallel)

    def get_batch_status(self, job: BatchJob) -> dict:
        """Get current status of a batch job.

        Args:
            job: Batch job to check

        Returns:
            Status dictionary
        """
        return {
            "job_name": job.name,
            "status": "in_progress" if job.pending_count > 0 else "complete",
            "progress": {
                "total": job.total_videos,
                "completed": job.completed_count,
                "failed": job.failed_count,
                "pending": job.pending_count,
                "percent": job.progress_percent,
            },
            "clips_generated": job.total_clips_generated,
            "failed_videos": [
                {"path": str(p), "error": e}
                for p, e in job.get_failed_videos()
            ],
        }
