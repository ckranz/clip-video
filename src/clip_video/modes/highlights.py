"""Highlights mode for creating social media shorts.

Implements the complete highlights workflow:
1. Load video and session description
2. Transcribe video if needed
3. Send transcript to LLM for highlight analysis
4. Extract highlight clips
5. Convert to portrait format
6. Burn captions with brand enhancements
7. Generate metadata for each platform
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from clip_video.captions.enhancements import BrandEnhancements, EnhancedCaptionRenderer
from clip_video.captions.renderer import Caption, CaptionTrack, CaptionRenderer, slice_text_into_phrases
from clip_video.captions.styles import CaptionStyle, YOUTUBE_SHORTS_STYLE
from clip_video.ffmpeg import FFmpegWrapper, ExtractionConfig, ClipPadding
from clip_video.llm.base import (
    LLMConfig,
    LLMProviderType,
    HighlightSegment,
    HighlightAnalysis,
)
from clip_video.llm.claude import ClaudeLLM
from clip_video.llm.openai import OpenAILLM
from clip_video.llm.ollama import OllamaLLM
from clip_video.validation.orchestrator import AgenticValidator, RunSummary
from clip_video.progress import ProgressTracker
from clip_video.state import ProcessingState, IdempotentProcessor
from clip_video.storage import atomic_write_json, read_json
from clip_video.transcription import TranscriptionSegment
from clip_video.video.portrait import (
    PortraitConfig,
    PortraitConverter,
    AspectRatio,
    LogoOverlayConfig,
    YOUTUBE_SHORTS_CONFIG,
    TIKTOK_CONFIG,
    INSTAGRAM_REELS_CONFIG,
)
from clip_video.config import BrandConfig, load_brand_config, get_brand_path, SocialCopyStyle
from clip_video.llm.social_copy import (
    SocialCopy,
    build_social_copy_prompt,
    parse_social_copy_response,
    save_social_copy,
)


class Platform:
    """Supported social media platforms."""

    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"


@dataclass
class HighlightClip:
    """A generated highlight clip.

    Attributes:
        clip_id: Unique identifier for this clip
        segment: The highlight segment this clip is from
        source_video: Path to the source video
        raw_clip_path: Path to raw extracted clip
        portrait_clip_path: Path to portrait-converted clip
        captioned_clip_path: Path to clip with captions
        metadata: Generated metadata for social media
        created_at: When the clip was created
    """

    clip_id: str
    segment: HighlightSegment
    source_video: Path
    raw_clip_path: Path | None = None
    portrait_clip_path: Path | None = None
    captioned_clip_path: Path | None = None
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def final_clip_path(self) -> Path | None:
        """Get the final processed clip path."""
        return self.captioned_clip_path or self.portrait_clip_path or self.raw_clip_path

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "segment": self.segment.to_dict(),
            "source_video": str(self.source_video),
            "raw_clip_path": str(self.raw_clip_path) if self.raw_clip_path else None,
            "portrait_clip_path": str(self.portrait_clip_path) if self.portrait_clip_path else None,
            "captioned_clip_path": str(self.captioned_clip_path) if self.captioned_clip_path else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HighlightClip":
        """Create from dictionary."""
        return cls(
            clip_id=data["clip_id"],
            segment=HighlightSegment.from_dict(data["segment"]),
            source_video=Path(data["source_video"]),
            raw_clip_path=Path(data["raw_clip_path"]) if data.get("raw_clip_path") else None,
            portrait_clip_path=Path(data["portrait_clip_path"]) if data.get("portrait_clip_path") else None,
            captioned_clip_path=Path(data["captioned_clip_path"]) if data.get("captioned_clip_path") else None,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )


@dataclass
class HighlightMetadata:
    """Generated metadata for a highlight clip.

    Attributes:
        title: Suggested title for the clip
        hook_text: Hook text for the first few seconds
        description: Longer description
        hashtags: Suggested hashtags
        platform_specific: Platform-specific metadata
    """

    title: str
    hook_text: str
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    platform_specific: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "hook_text": self.hook_text,
            "description": self.description,
            "hashtags": self.hashtags,
            "platform_specific": self.platform_specific,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HighlightMetadata":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            hook_text=data.get("hook_text", ""),
            description=data.get("description", ""),
            hashtags=data.get("hashtags", []),
            platform_specific=data.get("platform_specific", {}),
        )


@dataclass
class HighlightsConfig:
    """Configuration for highlights processing.

    Attributes:
        target_clips: Number of highlight clips to generate
        min_acceptable_clips: Minimum clips for a successful run
        min_duration: Minimum clip duration in seconds (YouTube Shorts: 30-120s)
        max_duration: Maximum clip duration in seconds (YouTube Shorts: 30-120s)
        max_replacement_attempts: Max attempts to replace a rejected clip
        cost_ceiling_gbp: Maximum LLM cost per video run in GBP
        enable_validation_pass: Enable multi-pass validation workflow
        output_format: Output video format
        portrait_config: Portrait conversion settings
        caption_style: Caption styling
        enhancements: Brand-specific enhancements
        platforms: Target platforms for output
        llm_config: LLM configuration for analysis
        clip_padding: Padding for clip extraction
        brand_config: Brand configuration for portrait/logo settings
    """

    target_clips: int = 5
    min_acceptable_clips: int = 3
    min_duration: float = 30.0  # YouTube Shorts compatible (was 15.0)
    max_duration: float = 120.0  # YouTube Shorts compatible (was 60.0)
    max_replacement_attempts: int = 3
    cost_ceiling_gbp: float = 5.0
    enable_validation_pass: bool = True
    output_format: str = "mp4"
    portrait_config: PortraitConfig = field(default_factory=lambda: YOUTUBE_SHORTS_CONFIG)
    caption_style: CaptionStyle = field(default_factory=lambda: YOUTUBE_SHORTS_STYLE)
    enhancements: BrandEnhancements | None = None
    platforms: list[str] = field(default_factory=lambda: [Platform.YOUTUBE_SHORTS])
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    clip_padding: ClipPadding = field(default_factory=ClipPadding)
    brand_config: BrandConfig | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_duration > self.max_duration:
            raise ValueError(
                f"min_duration ({self.min_duration}) cannot be greater than "
                f"max_duration ({self.max_duration})"
            )
        if self.min_acceptable_clips > self.target_clips:
            raise ValueError(
                f"min_acceptable_clips ({self.min_acceptable_clips}) cannot be greater than "
                f"target_clips ({self.target_clips})"
            )

    def get_platform_config(
        self,
        platform: str,
        source_width: int | None = None,
    ) -> PortraitConfig:
        """Get portrait config for a specific platform with brand overrides.

        Args:
            platform: Target platform (youtube_shorts, tiktok, etc.)
            source_width: Source video width for crop offset calculation

        Returns:
            PortraitConfig with brand-specific overrides applied
        """
        # Get base platform config
        configs = {
            Platform.YOUTUBE_SHORTS: YOUTUBE_SHORTS_CONFIG,
            Platform.TIKTOK: TIKTOK_CONFIG,
            Platform.INSTAGRAM_REELS: INSTAGRAM_REELS_CONFIG,
        }
        base_config = configs.get(platform, self.portrait_config)

        # If no brand config, return base config
        if not self.brand_config:
            return base_config

        # Apply brand-specific overrides
        crop_x_offset = base_config.crop_x_offset
        if source_width:
            crop_x_offset = self.brand_config.get_crop_x_offset(source_width)
        elif self.brand_config.portrait.crop_x_offset != 0.5:
            crop_x_offset = self.brand_config.portrait.crop_x_offset

        # Build logo config from brand settings
        logo_settings = self.brand_config.logo
        logo_config = LogoOverlayConfig(
            enabled=logo_settings.enabled,
            logo_path=get_brand_path(self.brand_config.name) / logo_settings.image_path if logo_settings.enabled else None,
            position=logo_settings.position,
            height_percent=logo_settings.height_percent,
            opacity=logo_settings.opacity,
            margin=logo_settings.margin,
        )

        # Create new config with overrides
        return PortraitConfig(
            target_ratio=base_config.target_ratio,
            target_width=base_config.target_width,
            crop_x_offset=crop_x_offset,
            crop_y_offset=base_config.crop_y_offset,
            video_codec=base_config.video_codec,
            video_crf=base_config.video_crf,
            video_preset=base_config.video_preset,
            audio_codec=base_config.audio_codec,
            audio_bitrate=base_config.audio_bitrate,
            faststart=base_config.faststart,
            logo=logo_config,
        )


@dataclass
class HighlightsProject:
    """A highlights project for generating social media clips.

    Tracks the state of a highlights project including:
    - Source video and description
    - Transcript and analysis
    - Generated clips and their status
    - Output metadata

    Attributes:
        name: Project name
        brand_name: Brand name
        video_path: Path to source video
        description_path: Path to session description file
        transcript_text: Full transcript text
        analysis: LLM highlight analysis results
        clips: Generated highlight clips
        config: Processing configuration
        created_at: When project was created
        updated_at: Last update time
    """

    name: str
    brand_name: str
    video_path: Path
    description_path: Path | None = None
    transcript_text: str = ""
    analysis: HighlightAnalysis | None = None
    clips: list[HighlightClip] = field(default_factory=list)
    config: HighlightsConfig = field(default_factory=HighlightsConfig)
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
        return Path.cwd() / "brands" / self.brand_name / "highlights" / self.name

    @project_root.setter
    def project_root(self, value: Path):
        """Set the project root directory."""
        self._project_root = value

    @property
    def clips_dir(self) -> Path:
        """Get the directory for extracted clips."""
        return self.project_root / "clips"

    @property
    def raw_clips_dir(self) -> Path:
        """Get the directory for raw extracted clips."""
        return self.clips_dir / "raw"

    @property
    def portrait_clips_dir(self) -> Path:
        """Get the directory for portrait clips."""
        return self.clips_dir / "portrait"

    @property
    def final_clips_dir(self) -> Path:
        """Get the directory for final captioned clips."""
        return self.clips_dir / "final"

    @property
    def metadata_dir(self) -> Path:
        """Get the directory for metadata files."""
        return self.project_root / "metadata"

    @property
    def state_file(self) -> Path:
        """Get the state file path."""
        return self.project_root / "project_state.json"

    def get_summary(self) -> dict:
        """Get project summary statistics."""
        return {
            "project_name": self.name,
            "brand_name": self.brand_name,
            "video_path": str(self.video_path),
            "description_path": str(self.description_path) if self.description_path else None,
            "has_transcript": bool(self.transcript_text),
            "has_analysis": self.analysis is not None,
            "segments_identified": len(self.analysis.segments) if self.analysis else 0,
            "clips_generated": len(self.clips),
            "clips_with_captions": sum(1 for c in self.clips if c.captioned_clip_path),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "brand_name": self.brand_name,
            "video_path": str(self.video_path),
            "description_path": str(self.description_path) if self.description_path else None,
            "transcript_text": self.transcript_text,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "clips": [c.to_dict() for c in self.clips],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict, config: HighlightsConfig | None = None) -> "HighlightsProject":
        """Create from dictionary."""
        project = cls(
            name=data["name"],
            brand_name=data["brand_name"],
            video_path=Path(data["video_path"]),
            description_path=Path(data["description_path"]) if data.get("description_path") else None,
            transcript_text=data.get("transcript_text", ""),
            analysis=HighlightAnalysis.from_dict(data["analysis"]) if data.get("analysis") else None,
            clips=[HighlightClip.from_dict(c) for c in data.get("clips", [])],
            config=config or HighlightsConfig(),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        return project

    def save(self) -> None:
        """Save project state to disk."""
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        atomic_write_json(self.state_file, self.to_dict())

    @classmethod
    def load(cls, state_file: Path, config: HighlightsConfig | None = None) -> "HighlightsProject":
        """Load project from state file."""
        data = read_json(state_file)
        project = cls.from_dict(data, config)
        project._project_root = state_file.parent
        return project


class HighlightsProcessor:
    """Processes videos to generate highlight clips.

    Orchestrates the full highlights workflow:
    1. Transcription (if needed)
    2. LLM analysis for highlight detection
    3. Clip extraction
    4. Portrait conversion
    5. Caption burning with enhancements
    6. Metadata generation
    """

    def __init__(
        self,
        config: HighlightsConfig | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        """Initialize the processor.

        Args:
            config: Processing configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or HighlightsConfig()
        self.progress_callback = progress_callback
        self.ffmpeg = FFmpegWrapper()
        self.portrait_converter = PortraitConverter()

        # Initialize LLM provider
        if self.config.llm_config.provider == LLMProviderType.CLAUDE:
            self.llm = ClaudeLLM(self.config.llm_config)
        elif self.config.llm_config.provider == LLMProviderType.OPENAI:
            self.llm = OpenAILLM(self.config.llm_config)
        elif self.config.llm_config.provider == LLMProviderType.OLLAMA:
            self.llm = OllamaLLM(self.config.llm_config)
        else:
            # Default to Claude
            self.llm = ClaudeLLM(self.config.llm_config)

    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, progress)

    def create_project(
        self,
        name: str,
        brand_name: str,
        video_path: Path,
        description_path: Path | None = None,
        project_root: Path | None = None,
    ) -> HighlightsProject:
        """Create a new highlights project.

        Args:
            name: Project name
            brand_name: Brand name
            video_path: Path to source video
            description_path: Optional path to description file
            project_root: Optional custom project root

        Returns:
            New HighlightsProject
        """
        project = HighlightsProject(
            name=name,
            brand_name=brand_name,
            video_path=video_path,
            description_path=description_path,
            config=self.config,
        )

        if project_root:
            project.project_root = project_root

        # Create directories
        project.project_root.mkdir(parents=True, exist_ok=True)
        project.raw_clips_dir.mkdir(parents=True, exist_ok=True)
        project.portrait_clips_dir.mkdir(parents=True, exist_ok=True)
        project.final_clips_dir.mkdir(parents=True, exist_ok=True)
        project.metadata_dir.mkdir(parents=True, exist_ok=True)

        project.save()
        return project

    def load_description(self, project: HighlightsProject) -> str:
        """Load session description from file.

        Args:
            project: The project to load description for

        Returns:
            Description text
        """
        if not project.description_path or not project.description_path.exists():
            return ""

        return project.description_path.read_text(encoding="utf-8")

    def transcribe(
        self,
        project: HighlightsProject,
        transcript_segments: list[TranscriptionSegment] | None = None,
    ) -> str:
        """Get or generate transcript for the video.

        Args:
            project: The project to transcribe
            transcript_segments: Optional pre-existing transcript segments

        Returns:
            Formatted transcript text with timestamps
        """
        self._report_progress("transcription", 0.0)

        if project.transcript_text:
            self._report_progress("transcription", 1.0)
            return project.transcript_text

        if transcript_segments:
            # Format segments into transcript text
            lines = []
            for seg in transcript_segments:
                timestamp = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
                lines.append(f"{timestamp} {seg.text}")

            project.transcript_text = "\n".join(lines)
            project.save()
            self._report_progress("transcription", 1.0)
            return project.transcript_text

        # For now, require transcript to be provided
        # In a full implementation, this would call the transcription service
        raise ValueError(
            "No transcript available. Please provide transcript_segments or "
            "ensure the video has been transcribed."
        )

    def analyze(self, project: HighlightsProject) -> HighlightAnalysis:
        """Analyze transcript to identify highlights.

        Args:
            project: Project with transcript to analyze

        Returns:
            HighlightAnalysis with identified segments
        """
        self._report_progress("analysis", 0.0)

        if project.analysis:
            self._report_progress("analysis", 1.0)
            return project.analysis

        if not project.transcript_text:
            raise ValueError("Project has no transcript. Call transcribe() first.")

        description = self.load_description(project)

        self._report_progress("analysis", 0.3)

        analysis = self.llm.analyze_transcript(
            transcript_text=project.transcript_text,
            session_description=description,
            target_clips=self.config.target_clips,
        )
        analysis.video_id = project.name

        # Filter segments by duration constraints
        valid_segments = [
            seg for seg in analysis.segments
            if self.config.min_duration <= seg.duration <= self.config.max_duration
        ]
        analysis.segments = valid_segments

        project.analysis = analysis
        project.save()

        self._report_progress("analysis", 1.0)
        return analysis

    def validate_and_refine(
        self,
        project: HighlightsProject,
        brand_context: dict | None = None,
    ) -> tuple[list[HighlightSegment], RunSummary]:
        """Run multi-pass validation on analyzed segments.

        Uses the AgenticValidator to:
        1. Validate each segment against quality criteria
        2. Reject failing segments to review queue
        3. Find replacement candidates for rejected segments
        4. Repeat until target reached or resources exhausted

        Args:
            project: Project with analysis to validate
            brand_context: Optional brand context for validation

        Returns:
            Tuple of (approved_segments, run_summary)
        """
        self._report_progress("validation", 0.0)

        if not project.analysis:
            raise ValueError("Project has no analysis. Call analyze() first.")

        if not self.config.enable_validation_pass:
            # Skip validation, return all segments as-is
            return project.analysis.segments, RunSummary(
                approved_count=len(project.analysis.segments),
                rejected_count=0,
                target_clips=self.config.target_clips,
                min_acceptable=self.config.min_acceptable_clips,
                target_met=len(project.analysis.segments) >= self.config.target_clips,
                minimum_met=len(project.analysis.segments) >= self.config.min_acceptable_clips,
                iterations=0,
                total_cost_usd=0.0,
                total_cost_gbp=0.0,
                cost_ceiling_gbp=self.config.cost_ceiling_gbp,
                cost_ceiling_hit=False,
                transcript_exhausted=False,
                termination_reason="Validation pass disabled",
            )

        self._report_progress("validation", 0.1)

        # Create validator
        validator = AgenticValidator(
            config=self.config,
            llm=self.llm,
            transcript_text=project.transcript_text,
            brand_context=brand_context or {},
            project_root=project.project_root,
            video_path=str(project.video_path),
        )

        self._report_progress("validation", 0.2)

        # Run validation
        approved_segments, summary = validator.validate_and_refine(
            initial_segments=project.analysis.segments,
        )

        # Update analysis with approved segments only
        project.analysis.segments = approved_segments
        project.save()

        self._report_progress("validation", 1.0)
        return approved_segments, summary

    def extract_clips(self, project: HighlightsProject) -> list[HighlightClip]:
        """Extract raw clips from source video.

        Args:
            project: Project with analysis to extract clips from

        Returns:
            List of HighlightClip objects with raw_clip_path set
        """
        self._report_progress("extraction", 0.0)

        if not project.analysis:
            raise ValueError("Project has no analysis. Call analyze() first.")

        if not project.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {project.video_path}")

        clips = []
        total_segments = len(project.analysis.segments)

        for i, segment in enumerate(project.analysis.segments):
            clip_id = f"clip_{i+1:02d}"

            # Check if clip already exists in project
            existing = next((c for c in project.clips if c.clip_id == clip_id), None)
            if existing and existing.raw_clip_path and existing.raw_clip_path.exists():
                clips.append(existing)
                self._report_progress("extraction", (i + 1) / total_segments)
                continue

            # Extract the clip
            output_path = project.raw_clips_dir / f"{clip_id}.{self.config.output_format}"

            extraction_config = ExtractionConfig(
                start_time=segment.start_time,
                end_time=segment.end_time,
                padding=self.config.clip_padding,
            )

            self.ffmpeg.extract_clip(
                input_path=project.video_path,
                output_path=output_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                config=extraction_config,
            )

            clip = HighlightClip(
                clip_id=clip_id,
                segment=segment,
                source_video=project.video_path,
                raw_clip_path=output_path,
            )
            clips.append(clip)

            self._report_progress("extraction", (i + 1) / total_segments)

        project.clips = clips
        project.save()
        return clips

    def convert_to_portrait(
        self,
        project: HighlightsProject,
        platform: str = Platform.YOUTUBE_SHORTS,
    ) -> list[HighlightClip]:
        """Convert clips to portrait format.

        Args:
            project: Project with extracted clips
            platform: Target platform for format settings

        Returns:
            Updated clips with portrait_clip_path set
        """
        self._report_progress("portrait_conversion", 0.0)

        if not project.clips:
            raise ValueError("Project has no clips. Call extract_clips() first.")

        total_clips = len(project.clips)

        for i, clip in enumerate(project.clips):
            if not clip.raw_clip_path or not clip.raw_clip_path.exists():
                continue

            if clip.portrait_clip_path and clip.portrait_clip_path.exists():
                self._report_progress("portrait_conversion", (i + 1) / total_clips)
                continue

            # Get source video width for crop offset calculation
            video_info = self.ffmpeg.get_video_info(clip.raw_clip_path)
            portrait_config = self.config.get_platform_config(
                platform, source_width=video_info.width
            )

            output_path = project.portrait_clips_dir / f"{clip.clip_id}_portrait.{self.config.output_format}"

            self.portrait_converter.convert(
                input_path=clip.raw_clip_path,
                output_path=output_path,
                config=portrait_config,
            )

            clip.portrait_clip_path = output_path
            self._report_progress("portrait_conversion", (i + 1) / total_clips)

        project.save()
        return project.clips

    def _build_caption_track(
        self,
        clip: HighlightClip,
        transcript_segments: list[TranscriptionSegment],
        max_words_per_phrase: int = 4,
    ) -> CaptionTrack:
        """Build caption track for a clip from transcript segments.

        Slices long transcript segments into short phrases for
        single-line centered captions (TikTok/Shorts style).

        Args:
            clip: The clip to build captions for
            transcript_segments: Full transcript segments
            max_words_per_phrase: Maximum words per caption phrase (default 4)

        Returns:
            CaptionTrack with phrase-sliced captions for this clip's time range
        """
        track = CaptionTrack(default_style=self.config.caption_style)

        # Find segments that overlap with the clip
        clip_start = clip.segment.start_time
        clip_end = clip.segment.end_time

        for seg in transcript_segments:
            # Check for overlap
            if seg.end < clip_start or seg.start > clip_end:
                continue

            # Adjust times relative to clip start
            relative_start = max(0, seg.start - clip_start)
            relative_end = min(clip_end - clip_start, seg.end - clip_start)

            # Slice into short phrases for single-line display
            phrases = slice_text_into_phrases(
                seg.text,
                relative_start,
                relative_end,
                max_words_per_phrase,
            )

            for phrase_text, phrase_start, phrase_end in phrases:
                track.add_caption(
                    text=phrase_text,
                    start_time=phrase_start,
                    end_time=phrase_end,
                )

        return track

    def burn_captions(
        self,
        project: HighlightsProject,
        transcript_segments: list[TranscriptionSegment] | None = None,
    ) -> list[HighlightClip]:
        """Burn captions into clips with brand enhancements.

        Args:
            project: Project with portrait clips
            transcript_segments: Optional transcript segments for caption timing

        Returns:
            Updated clips with captioned_clip_path set
        """
        self._report_progress("captions", 0.0)

        if not project.clips:
            raise ValueError("Project has no clips. Call extract_clips() first.")

        # Use enhanced renderer if enhancements are configured
        if self.config.enhancements:
            renderer = EnhancedCaptionRenderer()
        else:
            renderer = CaptionRenderer()

        total_clips = len(project.clips)

        for i, clip in enumerate(project.clips):
            # Get the source clip (portrait if available, otherwise raw)
            source_path = clip.portrait_clip_path or clip.raw_clip_path
            if not source_path or not source_path.exists():
                continue

            if clip.captioned_clip_path and clip.captioned_clip_path.exists():
                self._report_progress("captions", (i + 1) / total_clips)
                continue

            output_path = project.final_clips_dir / f"{clip.clip_id}_final.{self.config.output_format}"

            # Build caption track
            if transcript_segments:
                caption_track = self._build_caption_track(clip, transcript_segments)
            else:
                # Create minimal caption track with hook text
                caption_track = CaptionTrack(default_style=self.config.caption_style)
                if clip.segment.hook_text:
                    caption_track.add_caption(
                        text=clip.segment.hook_text,
                        start_time=0.0,
                        end_time=min(3.0, clip.segment.duration),
                    )

            # Render with or without enhancements
            if self.config.enhancements and isinstance(renderer, EnhancedCaptionRenderer):
                renderer.render_with_enhancements(
                    input_path=source_path,
                    output_path=output_path,
                    caption_track=caption_track,
                    enhancements=self.config.enhancements,
                )
            else:
                renderer.render(
                    input_path=source_path,
                    output_path=output_path,
                    caption_track=caption_track,
                )

            clip.captioned_clip_path = output_path
            self._report_progress("captions", (i + 1) / total_clips)

        project.save()
        return project.clips

    def generate_social_copy(
        self,
        project: HighlightsProject,
        transcript_segments: list[TranscriptionSegment] | None = None,
    ) -> list[SocialCopy]:
        """Generate voice-styled social media copy for each clip.

        Creates a markdown file alongside each final clip with hook,
        description, and hashtags styled according to brand settings.

        Args:
            project: Project with clips
            transcript_segments: Optional transcript for clip text

        Returns:
            List of SocialCopy objects
        """
        self._report_progress("social_copy", 0.0)

        if not project.clips:
            return []

        # Check if social copy is enabled
        if not self.config.brand_config or not self.config.brand_config.social_copy.enabled:
            return []

        style = self.config.brand_config.social_copy
        social_copies = []
        total_clips = len(project.clips)

        # Extract speaker name and talk title from project name
        # Format is typically "Speaker Name - Talk Title_YYYYMMDD_HHMMSS"
        speaker_name = ""
        talk_title = ""
        if " - " in project.name:
            parts = project.name.split(" - ", 1)
            speaker_name = parts[0].strip()
            # Remove timestamp suffix from talk title (_YYYYMMDD_HHMMSS)
            talk_part = parts[1] if len(parts) > 1 else ""
            # Look for _YYYYMMDD_HHMMSS pattern and remove it
            timestamp_match = re.search(r"_\d{8}_\d{6}$", talk_part)
            if timestamp_match:
                talk_title = talk_part[:timestamp_match.start()].strip()
            elif "_" in talk_part:
                talk_title = talk_part.rsplit("_", 1)[0].strip()
            else:
                talk_title = talk_part.strip()

        for i, clip in enumerate(project.clips):
            segment = clip.segment

            # Get clip transcript text
            clip_transcript = ""
            if transcript_segments:
                # Find segments that overlap with this clip
                clip_segs = [
                    seg for seg in transcript_segments
                    if seg.end > segment.start_time and seg.start < segment.end_time
                ]
                clip_transcript = " ".join(seg.text for seg in clip_segs)

            # Build prompts
            system_prompt, user_prompt = build_social_copy_prompt(
                style=style,
                clip_summary=segment.summary,
                clip_transcript=clip_transcript or segment.summary,
                speaker_name=speaker_name,
                talk_title=talk_title,
                topics=segment.topics,
            )

            # Generate with LLM
            try:
                client = self.llm._get_client()
                response = client.messages.create(
                    model=self.config.llm_config.model,
                    max_tokens=500,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                response_text = response.content[0].text
            except Exception as e:
                # Fallback to existing hook_text and summary
                response_text = json.dumps({
                    "hook": segment.hook_text or segment.summary[:100],
                    "description": segment.summary,
                    "hashtags": [t.replace(" ", "") for t in segment.topics[:5]],
                })

            # Parse response
            social_copy = parse_social_copy_response(
                response_text=response_text,
                style=style,
                speaker_name=speaker_name,
                talk_title=talk_title,
                topics=segment.topics,
            )
            social_copies.append(social_copy)

            # Save markdown file alongside final clip
            if clip.captioned_clip_path:
                md_path = clip.captioned_clip_path.with_suffix(".md")
                video_filename = clip.captioned_clip_path.name
            else:
                md_path = project.final_clips_dir / f"{clip.clip_id}_final.md"
                video_filename = f"{clip.clip_id}_final.{self.config.output_format}"

            save_social_copy(social_copy, md_path, video_filename)

            self._report_progress("social_copy", (i + 1) / total_clips)

        return social_copies

    def generate_metadata(self, project: HighlightsProject) -> list[dict]:
        """Generate metadata files for each clip.

        Args:
            project: Project with generated clips

        Returns:
            List of metadata dictionaries
        """
        self._report_progress("metadata", 0.0)

        if not project.clips:
            raise ValueError("Project has no clips.")

        metadata_list = []
        total_clips = len(project.clips)

        for i, clip in enumerate(project.clips):
            metadata = self._generate_clip_metadata(clip, project)
            clip.metadata = metadata

            # Write metadata file
            metadata_path = project.metadata_dir / f"{clip.clip_id}_metadata.json"
            atomic_write_json(metadata_path, metadata)

            metadata_list.append(metadata)
            self._report_progress("metadata", (i + 1) / total_clips)

        project.save()
        return metadata_list

    def _generate_clip_metadata(
        self,
        clip: HighlightClip,
        project: HighlightsProject,
    ) -> dict:
        """Generate metadata for a single clip.

        Args:
            clip: The clip to generate metadata for
            project: The parent project

        Returns:
            Metadata dictionary
        """
        segment = clip.segment

        # Generate hashtags from topics
        hashtags = [f"#{topic.replace(' ', '')}" for topic in segment.topics[:5]]

        # Platform-specific metadata
        platform_specific = {}

        if Platform.YOUTUBE_SHORTS in self.config.platforms:
            platform_specific["youtube_shorts"] = {
                "title": segment.summary[:100] if segment.summary else "",
                "description": f"{segment.hook_text}\n\n{segment.reason}",
                "tags": segment.topics,
            }

        if Platform.TIKTOK in self.config.platforms:
            platform_specific["tiktok"] = {
                "caption": f"{segment.hook_text} {' '.join(hashtags[:5])}",
            }

        if Platform.INSTAGRAM_REELS in self.config.platforms:
            platform_specific["instagram_reels"] = {
                "caption": f"{segment.hook_text}\n.\n.\n{' '.join(hashtags)}",
            }

        return {
            "clip_id": clip.clip_id,
            "title": segment.summary,
            "hook_text": segment.hook_text,
            "description": segment.reason,
            "topics": segment.topics,
            "hashtags": hashtags,
            "duration": segment.duration,
            "quality_score": segment.quality_score,
            "source_video": str(project.video_path.name),
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "platform_specific": platform_specific,
            "output_files": {
                "raw": str(clip.raw_clip_path) if clip.raw_clip_path else None,
                "portrait": str(clip.portrait_clip_path) if clip.portrait_clip_path else None,
                "final": str(clip.captioned_clip_path) if clip.captioned_clip_path else None,
            },
        }

    def process(
        self,
        project: HighlightsProject,
        transcript_segments: list[TranscriptionSegment] | None = None,
        skip_captions: bool = False,
        brand_context: dict | None = None,
        skip_validation: bool = False,
    ) -> tuple[HighlightsProject, RunSummary | None]:
        """Run the full highlights workflow.

        Args:
            project: Project to process
            transcript_segments: Optional transcript segments
            skip_captions: Skip caption burning step
            brand_context: Optional brand context for validation
            skip_validation: Skip the validation pass

        Returns:
            Tuple of (updated project, validation summary or None)
        """
        # Step 1: Ensure we have a transcript
        self.transcribe(project, transcript_segments)

        # Step 2: Analyze for highlights
        self.analyze(project)

        # Step 3: Optional validation pass
        validation_summary = None
        if self.config.enable_validation_pass and not skip_validation:
            _, validation_summary = self.validate_and_refine(project, brand_context)

        # Step 4: Extract clips (from approved segments)
        self.extract_clips(project)

        # Step 5: Convert to portrait
        self.convert_to_portrait(project)

        # Step 6: Burn captions (unless skipped)
        if not skip_captions:
            self.burn_captions(project, transcript_segments)

        # Step 7: Generate metadata
        self.generate_metadata(project)

        return project, validation_summary

    def get_cost_estimate(self, transcript_text: str) -> float:
        """Estimate the cost of processing a transcript.

        Args:
            transcript_text: Transcript to estimate

        Returns:
            Estimated cost in USD
        """
        return self.llm.estimate_cost(transcript_text)
