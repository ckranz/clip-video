"""Lyric Match mode for creating music video mashups.

Implements the full lyric match workflow:
1. Parse lyrics file into words/phrases
2. Search for each word/phrase across brand's video library
3. Extract candidate clips for each lyric line
4. Output organized clips for video editing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Callable

from clip_video.ffmpeg import FFmpegWrapper, ExtractionConfig, ClipPadding
from clip_video.lyrics.parser import LyricsParser, ParsedLyrics
from clip_video.lyrics.phrases import PhraseExtractor, ExtractionList, ExtractionTarget
from clip_video.search import BrandSearcher, SearchResult, SearchResults
from clip_video.selection import SelectionTracker, SelectionManager, Selection
from clip_video.storage import atomic_write_json, read_json
from clip_video.state import ProcessingState, IdempotentProcessor
from clip_video.progress import ProgressTracker


@dataclass
class LyricMatchConfig:
    """Configuration for lyric match processing.

    Attributes:
        max_candidates_per_target: Maximum clips to extract per word/phrase
        prefer_diversity: Prefer clips from different source videos
        extract_words: Whether to search for individual words
        extract_phrases: Whether to search for phrases
        clip_padding: Padding to add before/after clips
        output_format: Video output format
    """

    max_candidates_per_target: int = 5
    prefer_diversity: bool = True
    extract_words: bool = True
    extract_phrases: bool = True
    clip_padding: ClipPadding = field(default_factory=ClipPadding)
    output_format: str = "mp4"
    min_word_length: int = 2
    use_stop_words: bool = True


@dataclass
class LineClipSet:
    """Collection of candidate clips for a lyric line.

    Attributes:
        line_number: Line number in the lyrics
        line_text: Original text of the lyric line
        targets: Extraction targets for this line
        clips: Dict mapping target text to list of extracted clip paths
        coverage: Percentage of targets with at least one clip
    """

    line_number: int
    line_text: str
    targets: list[ExtractionTarget] = field(default_factory=list)
    clips: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Calculate what percentage of targets have clips."""
        if not self.targets:
            return 100.0

        covered = sum(1 for t in self.targets if t.text in self.clips and self.clips[t.text])
        return (covered / len(self.targets)) * 100

    @property
    def total_clips(self) -> int:
        """Total number of clips across all targets."""
        return sum(len(clips) for clips in self.clips.values())


@dataclass
class LyricMatchProject:
    """A lyric match project for creating music video mashups.

    Tracks the state of a lyric match project including:
    - Original lyrics and extraction targets
    - Search results for each target
    - Selected clips and extraction status
    - Coverage and gap reporting

    Attributes:
        name: Project name
        brand_name: Brand to search
        lyrics_file: Path to source lyrics file
        extraction_list: Parsed extraction targets
        line_clip_sets: Clips organized by lyric line
        created_at: When project was created
        updated_at: Last update time
    """

    name: str
    brand_name: str
    lyrics_file: Path
    extraction_list: ExtractionList | None = None
    line_clip_sets: list[LineClipSet] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    _project_root: Path | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        if self._project_root:
            return self._project_root
        return Path.cwd() / "brands" / self.brand_name / "projects" / self.name

    @project_root.setter
    def project_root(self, value: Path):
        """Set the project root directory."""
        self._project_root = value

    @property
    def clips_dir(self) -> Path:
        """Get the directory for extracted clips."""
        return self.project_root / "clips"

    @property
    def output_dir(self) -> Path:
        """Get the output directory for the project."""
        return self.project_root / "output"

    @property
    def total_targets(self) -> int:
        """Total number of extraction targets."""
        if self.extraction_list:
            return len(self.extraction_list.all_targets)
        return 0

    @property
    def coverage_percent(self) -> float:
        """Overall coverage percentage."""
        if not self.line_clip_sets:
            return 0.0

        total_targets = sum(len(lcs.targets) for lcs in self.line_clip_sets)
        if total_targets == 0:
            return 100.0

        covered = sum(
            1 for lcs in self.line_clip_sets
            for t in lcs.targets
            if t.text in lcs.clips and lcs.clips[t.text]
        )
        return (covered / total_targets) * 100

    def get_coverage_gaps(self) -> list[dict]:
        """Get list of targets missing clips.

        Returns:
            List of dicts with target info for missing clips
        """
        gaps = []
        for lcs in self.line_clip_sets:
            for target in lcs.targets:
                if target.text not in lcs.clips or not lcs.clips[target.text]:
                    gaps.append({
                        "line_number": lcs.line_number,
                        "line_text": lcs.line_text,
                        "target_text": target.text,
                        "is_phrase": target.is_phrase,
                    })
        return gaps

    def get_summary(self) -> dict:
        """Get project summary statistics.

        Returns:
            Dict with summary information
        """
        total_clips = sum(lcs.total_clips for lcs in self.line_clip_sets)
        gaps = self.get_coverage_gaps()

        return {
            "project_name": self.name,
            "brand_name": self.brand_name,
            "lyrics_file": str(self.lyrics_file),
            "total_lines": len(self.line_clip_sets),
            "total_targets": self.total_targets,
            "total_clips_extracted": total_clips,
            "coverage_percent": round(self.coverage_percent, 1),
            "missing_targets": len(gaps),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "brand_name": self.brand_name,
            "lyrics_file": str(self.lyrics_file),
            "extraction_list": self.extraction_list.to_dict() if self.extraction_list else None,
            "line_clip_sets": [
                {
                    "line_number": lcs.line_number,
                    "line_text": lcs.line_text,
                    "targets": [t.to_dict() for t in lcs.targets],
                    "clips": {k: [str(p) for p in v] for k, v in lcs.clips.items()},
                }
                for lcs in self.line_clip_sets
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LyricMatchProject":
        """Create from dictionary."""
        project = cls(
            name=data["name"],
            brand_name=data["brand_name"],
            lyrics_file=Path(data["lyrics_file"]),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

        if data.get("extraction_list"):
            project.extraction_list = ExtractionList.from_dict(data["extraction_list"])

        for lcs_data in data.get("line_clip_sets", []):
            lcs = LineClipSet(
                line_number=lcs_data["line_number"],
                line_text=lcs_data["line_text"],
                targets=[ExtractionTarget.from_dict(t) for t in lcs_data.get("targets", [])],
                clips={k: [Path(p) for p in v] for k, v in lcs_data.get("clips", {}).items()},
            )
            project.line_clip_sets.append(lcs)

        return project

    def save(self, path: Path | None = None) -> Path:
        """Save project to file.

        Args:
            path: Optional custom path, defaults to project_root/project.json

        Returns:
            Path to saved file
        """
        if path is None:
            path = self.project_root / "project.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.updated_at = datetime.now().isoformat()
        atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def load(cls, path: Path) -> "LyricMatchProject":
        """Load project from file.

        Args:
            path: Path to project file

        Returns:
            LyricMatchProject instance
        """
        data = read_json(path)
        project = cls.from_dict(data)
        project._project_root = path.parent
        return project


class LyricMatchProcessor:
    """Processor for lyric match projects.

    Handles the complete workflow from lyrics parsing to clip extraction.

    Example usage:
        processor = LyricMatchProcessor("my_brand")
        project = processor.create_project("my_song", Path("lyrics.txt"))
        processor.search_all(project)
        processor.extract_all(project)
    """

    def __init__(
        self,
        brand_name: str,
        brands_root: Path | None = None,
        config: LyricMatchConfig | None = None,
    ):
        """Initialize the processor.

        Args:
            brand_name: Brand to search
            brands_root: Root directory for brands
            config: Processing configuration
        """
        self.brand_name = brand_name
        self.brands_root = brands_root or Path.cwd() / "brands"
        self.config = config or LyricMatchConfig()

        self.brand_path = self.brands_root / brand_name
        self.searcher = BrandSearcher(brand_name, self.brands_root)
        self.selection_manager = SelectionManager(self.brands_root)
        self.ffmpeg = FFmpegWrapper()

    def create_project(
        self,
        project_name: str,
        lyrics_file: Path,
    ) -> LyricMatchProject:
        """Create a new lyric match project.

        Args:
            project_name: Name for the project
            lyrics_file: Path to lyrics file

        Returns:
            LyricMatchProject instance
        """
        # Parse lyrics
        parser = LyricsParser()
        lyrics = parser.parse_file(lyrics_file)

        # Extract targets
        extractor = PhraseExtractor(
            extract_words=self.config.extract_words,
            extract_phrases=self.config.extract_phrases,
            min_word_length=self.config.min_word_length,
            use_stop_words=self.config.use_stop_words,
        )
        extraction_list = extractor.extract(lyrics)

        # Create project
        project = LyricMatchProject(
            name=project_name,
            brand_name=self.brand_name,
            lyrics_file=lyrics_file.absolute(),
            extraction_list=extraction_list,
        )
        project._project_root = (
            self.brands_root / self.brand_name / "projects" / project_name
        )

        # Initialize line clip sets
        for line_num, targets in extraction_list.lines_in_order:
            # Find original line text
            line_text = ""
            for line in lyrics.content_lines:
                if line.line_number == line_num:
                    line_text = line.raw_text.strip()
                    break

            project.line_clip_sets.append(LineClipSet(
                line_number=line_num,
                line_text=line_text,
                targets=list(targets),
            ))

        # Save project
        project.save()

        return project

    def load_project(self, project_name: str) -> LyricMatchProject | None:
        """Load an existing project.

        Args:
            project_name: Name of the project

        Returns:
            LyricMatchProject if found, None otherwise
        """
        project_path = (
            self.brands_root / self.brand_name / "projects" / project_name / "project.json"
        )
        if not project_path.exists():
            return None
        return LyricMatchProject.load(project_path)

    def search_all(
        self,
        project: LyricMatchProject,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, SearchResults]:
        """Search for all targets in the project.

        Args:
            project: Project to search for
            progress_callback: Optional callback(target, current, total)

        Returns:
            Dict mapping target text to SearchResults
        """
        if not project.extraction_list:
            return {}

        all_results: dict[str, SearchResults] = {}
        targets = project.extraction_list.all_targets
        total = len(targets)

        for i, target in enumerate(targets):
            if progress_callback:
                progress_callback(target.text, i + 1, total)

            # Skip if already searched
            if target.text in all_results:
                continue

            # Search for target
            results = self.searcher.search(
                target.text,
                max_results=self.config.max_candidates_per_target * 2,
            )
            all_results[target.text] = results

            # Also search alternatives
            for alt in target.alternatives:
                if alt not in all_results:
                    alt_results = self.searcher.search(
                        alt,
                        max_results=self.config.max_candidates_per_target,
                    )
                    all_results[alt] = alt_results

        return all_results

    def extract_candidates(
        self,
        project: LyricMatchProject,
        search_results: dict[str, SearchResults],
        video_paths: dict[str, Path],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> int:
        """Extract candidate clips for all targets.

        Args:
            project: Project to extract clips for
            search_results: Search results from search_all
            video_paths: Dict mapping video_id to video file path
            progress_callback: Optional callback(target, current, total)

        Returns:
            Total number of clips extracted
        """
        total_extracted = 0
        all_targets = list(project.extraction_list.all_targets) if project.extraction_list else []
        total = len(all_targets)

        # Create clips directory
        clips_dir = project.clips_dir
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Create extraction config
        extraction_config = ExtractionConfig(
            padding=self.config.clip_padding,
        )

        for i, target in enumerate(all_targets):
            if progress_callback:
                progress_callback(target.text, i + 1, total)

            # Get search results for this target
            results = search_results.get(target.text)
            if not results or not results.results:
                continue

            # Find the line clip set for this target
            lcs = None
            for line_clip_set in project.line_clip_sets:
                for t in line_clip_set.targets:
                    if t.text == target.text:
                        lcs = line_clip_set
                        break
                if lcs:
                    break

            if not lcs:
                continue

            # Extract up to max_candidates clips
            extracted_paths = []
            for result in results.top_results(self.config.max_candidates_per_target):
                # Get video path
                video_path = video_paths.get(result.video_id)
                if not video_path or not video_path.exists():
                    continue

                # Generate output filename
                safe_target = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in target.text
                )[:30]
                clip_filename = (
                    f"line{lcs.line_number:03d}_{safe_target}_"
                    f"{result.video_id}_{result.start:.1f}s.{self.config.output_format}"
                )
                output_path = clips_dir / f"line_{lcs.line_number:03d}" / clip_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip if already extracted
                if output_path.exists():
                    extracted_paths.append(output_path)
                    continue

                # Extract clip
                try:
                    self.ffmpeg.extract_clip(
                        input_path=video_path,
                        output_path=output_path,
                        start_time=result.start,
                        end_time=result.end,
                        config=extraction_config,
                    )
                    extracted_paths.append(output_path)
                    total_extracted += 1
                except Exception as e:
                    # Log error but continue
                    print(f"Error extracting clip for '{target.text}': {e}")

            # Update line clip set
            if extracted_paths:
                lcs.clips[target.text] = extracted_paths

        # Save project state
        project.save()

        return total_extracted

    def get_video_paths(self, project: LyricMatchProject) -> dict[str, Path]:
        """Get paths to all video files for the brand.

        Args:
            project: Project to get video paths for

        Returns:
            Dict mapping video_id to video file path
        """
        video_paths = {}

        # Look in brand's videos directory
        videos_dir = self.brand_path / "videos"
        if videos_dir.exists():
            for video_file in videos_dir.rglob("*"):
                if video_file.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
                    video_id = video_file.stem
                    video_paths[video_id] = video_file

        # Also check projects directory
        projects_dir = self.brand_path / "projects"
        if projects_dir.exists():
            for video_file in projects_dir.rglob("*.mp4"):
                video_id = video_file.stem
                if video_id not in video_paths:
                    video_paths[video_id] = video_file

        return video_paths

    def generate_report(self, project: LyricMatchProject) -> str:
        """Generate a text report of the project status.

        Args:
            project: Project to report on

        Returns:
            Formatted report string
        """
        lines = []
        lines.append(f"# Lyric Match Project: {project.name}")
        lines.append(f"Brand: {project.brand_name}")
        lines.append(f"Lyrics: {project.lyrics_file.name}")
        lines.append("")

        summary = project.get_summary()
        lines.append("## Summary")
        lines.append(f"- Total lines: {summary['total_lines']}")
        lines.append(f"- Total targets: {summary['total_targets']}")
        lines.append(f"- Clips extracted: {summary['total_clips_extracted']}")
        lines.append(f"- Coverage: {summary['coverage_percent']}%")
        lines.append("")

        # Coverage by line
        lines.append("## Coverage by Line")
        for lcs in project.line_clip_sets:
            status = "✓" if lcs.coverage == 100 else "○" if lcs.coverage > 0 else "✗"
            lines.append(
                f"{status} Line {lcs.line_number}: {lcs.line_text[:50]}... "
                f"({lcs.total_clips} clips, {lcs.coverage:.0f}%)"
            )
        lines.append("")

        # Missing targets
        gaps = project.get_coverage_gaps()
        if gaps:
            lines.append("## Missing Targets")
            for gap in gaps:
                lines.append(
                    f"- Line {gap['line_number']}: \"{gap['target_text']}\" "
                    f"({'phrase' if gap['is_phrase'] else 'word'})"
                )
        else:
            lines.append("## All targets covered!")

        return "\n".join(lines)

    def save_report(self, project: LyricMatchProject) -> Path:
        """Save a report file for the project.

        Args:
            project: Project to report on

        Returns:
            Path to saved report
        """
        report = self.generate_report(project)
        report_path = project.project_root / "coverage_report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path
