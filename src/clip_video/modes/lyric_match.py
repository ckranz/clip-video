"""Lyric Match mode for creating music video mashups.

Implements the full lyric match workflow:
1. Parse lyrics file into words/phrases
2. Search for each word/phrase across brand's video library
3. Extract candidate clips for each lyric line
4. Output organized clips for video editing
"""

from __future__ import annotations

import json
import random
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
        shuffle_candidates: Shuffle candidates before limiting (for variety)
    """

    max_candidates_per_target: int = 10
    prefer_diversity: bool = True
    extract_words: bool = True
    extract_phrases: bool = True
    clip_padding: ClipPadding = field(default_factory=ClipPadding)
    output_format: str = "mp4"
    min_word_length: int = 1
    use_stop_words: bool = False  # For lyric matching, we want ALL words
    min_phrase_words: int = 2
    max_phrase_words: int = 5
    shuffle_candidates: bool = True  # Shuffle to get variety across videos


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
        """Overall coverage percentage (all targets)."""
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

    def get_word_coverage(self) -> tuple[int, int, list[str]]:
        """Get word coverage stats.

        Returns:
            Tuple of (words_found, total_words, missing_words)
        """
        total_words = 0
        found_words = 0
        missing_words = []

        for lcs in self.line_clip_sets:
            for target in lcs.targets:
                if not target.is_phrase:  # Single word
                    total_words += 1
                    if target.text in lcs.clips and lcs.clips[target.text]:
                        found_words += 1
                    else:
                        missing_words.append(target.text)

        # Deduplicate missing words while preserving order
        seen = set()
        unique_missing = []
        for word in missing_words:
            if word not in seen:
                seen.add(word)
                unique_missing.append(word)

        return found_words, total_words, unique_missing

    def get_phrase_coverage(self) -> tuple[int, int]:
        """Get phrase coverage stats.

        Returns:
            Tuple of (phrases_found, total_phrases)
        """
        total_phrases = 0
        found_phrases = 0

        for lcs in self.line_clip_sets:
            for target in lcs.targets:
                if target.is_phrase:  # Multi-word phrase
                    total_phrases += 1
                    if target.text in lcs.clips and lcs.clips[target.text]:
                        found_phrases += 1

        return found_phrases, total_phrases

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
        # Parse lyrics with phrase constraints
        parser = LyricsParser(
            min_phrase_words=self.config.min_phrase_words,
            max_phrase_words=self.config.max_phrase_words,
            generate_subphrases=True,  # Generate sliding window subphrases
        )
        lyrics = parser.parse_file(lyrics_file)

        # Extract targets
        extractor = PhraseExtractor(
            extract_words=self.config.extract_words,
            extract_phrases=self.config.extract_phrases,
            min_word_length=self.config.min_word_length,
            use_stop_words=self.config.use_stop_words,
            min_phrase_words=self.config.min_phrase_words,
            max_phrase_words=self.config.max_phrase_words,
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

    def update_project(
        self,
        project: LyricMatchProject,
        lyrics_file: Path,
    ) -> tuple[list[ExtractionTarget], list[ExtractionTarget]]:
        """Update an existing project with new lyrics.

        Re-parses the lyrics file and merges with existing project state.
        Preserves existing clips for unchanged targets, only adds new targets.

        Args:
            project: Existing project to update
            lyrics_file: Path to updated lyrics file

        Returns:
            Tuple of (new_targets, removed_targets)
        """
        # Parse new lyrics
        parser = LyricsParser(
            min_phrase_words=self.config.min_phrase_words,
            max_phrase_words=self.config.max_phrase_words,
            generate_subphrases=True,
        )
        lyrics = parser.parse_file(lyrics_file)

        # Extract new targets
        extractor = PhraseExtractor(
            extract_words=self.config.extract_words,
            extract_phrases=self.config.extract_phrases,
            min_word_length=self.config.min_word_length,
            use_stop_words=self.config.use_stop_words,
            min_phrase_words=self.config.min_phrase_words,
            max_phrase_words=self.config.max_phrase_words,
        )
        new_extraction_list = extractor.extract(lyrics)

        # Get old and new target texts
        old_targets = set()
        for lcs in project.line_clip_sets:
            for t in lcs.targets:
                old_targets.add(t.text)

        new_targets_set = set()
        for t in new_extraction_list.all_targets:
            new_targets_set.add(t.text)

        # Find additions and removals
        added_texts = new_targets_set - old_targets
        removed_texts = old_targets - new_targets_set

        # Build mapping of existing clips by target text
        existing_clips: dict[str, list[Path]] = {}
        for lcs in project.line_clip_sets:
            for target_text, clip_paths in lcs.clips.items():
                if target_text not in existing_clips:
                    existing_clips[target_text] = []
                existing_clips[target_text].extend(clip_paths)

        # Update project with new extraction list
        project.extraction_list = new_extraction_list
        project.lyrics_file = lyrics_file.absolute()

        # Rebuild line clip sets preserving existing clips
        project.line_clip_sets = []
        for line_num, targets in new_extraction_list.lines_in_order:
            # Find original line text
            line_text = ""
            for line in lyrics.content_lines:
                if line.line_number == line_num:
                    line_text = line.raw_text.strip()
                    break

            # Create line clip set
            lcs = LineClipSet(
                line_number=line_num,
                line_text=line_text,
                targets=list(targets),
            )

            # Restore existing clips for targets that still exist
            for target in targets:
                if target.text in existing_clips:
                    # Filter to only clips that still exist on disk
                    valid_clips = [p for p in existing_clips[target.text] if p.exists()]
                    if valid_clips:
                        lcs.clips[target.text] = valid_clips

            project.line_clip_sets.append(lcs)

        # Save updated project
        project.save()

        # Return lists of actual target objects
        added = [t for t in new_extraction_list.all_targets if t.text in added_texts]
        removed = [ExtractionTarget(text=t, source_line=0, source_text="") for t in removed_texts]

        return added, removed

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

            # Search for target - get more than needed so we can shuffle
            results = self.searcher.search(
                target.text,
                max_results=self.config.max_candidates_per_target * 5,
            )

            # Shuffle and limit results for variety across videos
            if self.config.shuffle_candidates and results.results:
                shuffled = list(results.results)
                random.shuffle(shuffled)
                results.results = shuffled[:self.config.max_candidates_per_target]
                results.total_count = len(results.results)

            all_results[target.text] = results

            # Also search alternatives
            for alt in target.alternatives:
                if alt not in all_results:
                    alt_results = self.searcher.search(
                        alt,
                        max_results=self.config.max_candidates_per_target * 5,
                    )
                    # Shuffle alternatives too
                    if self.config.shuffle_candidates and alt_results.results:
                        shuffled = list(alt_results.results)
                        random.shuffle(shuffled)
                        alt_results.results = shuffled[:self.config.max_candidates_per_target]
                        alt_results.total_count = len(alt_results.results)
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

                # Generate output filename - one folder per word/phrase
                safe_target = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in target.text
                )[:50]

                # Sanitize video_id for filename
                safe_video_id = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in result.video_id
                )[:40]

                clip_filename = (
                    f"{safe_video_id}_{result.start:.1f}s.{self.config.output_format}"
                )

                # Flat structure: clips/{word_or_phrase}/
                target_dir = clips_dir / safe_target
                output_path = target_dir / clip_filename
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

        # Get word and phrase coverage
        words_found, total_words, missing_words = project.get_word_coverage()
        phrases_found, total_phrases = project.get_phrase_coverage()

        word_coverage_pct = (words_found / total_words * 100) if total_words > 0 else 100.0
        phrase_coverage_pct = (phrases_found / total_phrases * 100) if total_phrases > 0 else 100.0

        summary = project.get_summary()
        lines.append("## Summary")
        lines.append(f"- Total lines: {summary['total_lines']}")
        lines.append(f"- **Word coverage: {words_found}/{total_words} ({word_coverage_pct:.0f}%)**")
        lines.append(f"- Phrase coverage: {phrases_found}/{total_phrases} ({phrase_coverage_pct:.0f}%)")
        lines.append(f"- Total clips extracted: {summary['total_clips_extracted']}")
        lines.append("")

        # Missing words (critical)
        if missing_words:
            lines.append("## ⚠️ Missing Words (Action Required)")
            lines.append("")
            lines.append("These words need lyrics changes, more source material, or manual recording:")
            lines.append("")
            for word in missing_words:
                lines.append(f"- \"{word}\"")
            lines.append("")

        # Coverage by line
        lines.append("## Coverage by Line")
        for lcs in project.line_clip_sets:
            # Count words vs phrases for this line
            line_words = [t for t in lcs.targets if not t.is_phrase]
            line_phrases = [t for t in lcs.targets if t.is_phrase]
            words_with_clips = sum(1 for t in line_words if t.text in lcs.clips and lcs.clips[t.text])

            word_status = "✓" if words_with_clips == len(line_words) else "○" if words_with_clips > 0 else "✗"
            lines.append(
                f"{word_status} Line {lcs.line_number}: {lcs.line_text[:50]}{'...' if len(lcs.line_text) > 50 else ''} "
                f"(words: {words_with_clips}/{len(line_words)}, phrases: {len([t for t in line_phrases if t.text in lcs.clips and lcs.clips[t.text]])}/{len(line_phrases)})"
            )
        lines.append("")

        # Missing phrases (informational)
        gaps = project.get_coverage_gaps()
        missing_phrase_gaps = [g for g in gaps if g['is_phrase']]
        if missing_phrase_gaps:
            lines.append(f"## Missing Phrases ({len(missing_phrase_gaps)} total)")
            lines.append("")
            lines.append("Phrases are nice-to-have but not required - you can build them from individual words.")
            lines.append("")
            # Only show first 20 to avoid overwhelming the report
            for gap in missing_phrase_gaps[:20]:
                lines.append(f"- \"{gap['target_text']}\"")
            if len(missing_phrase_gaps) > 20:
                lines.append(f"- ... and {len(missing_phrase_gaps) - 20} more")

        if not missing_words and not missing_phrase_gaps:
            lines.append("## ✓ All targets covered!")

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
