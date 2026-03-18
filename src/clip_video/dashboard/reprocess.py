from __future__ import annotations

from pathlib import Path
from typing import Callable

from clip_video.catalogue import load_catalogue
from clip_video.captions.renderer import CaptionRenderer, CaptionTrack
from clip_video.modes.highlights import HighlightsProject
from clip_video.video.portrait import PortraitConfig, PortraitConverter

# Maps speaker position to crop_x_offset (0.0=left edge, 1.0=right edge)
POSITION_TO_OFFSET = {
    "left": 0.25,
    "center": 0.5,
    "right": 0.75,
}


def reprocess_video_clips(
    project: HighlightsProject,
    catalogue_path: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Re-crop and re-burn captions for all clips using updated catalogue data."""
    catalogue = load_catalogue(catalogue_path)
    video_filename = Path(project.video_path).name
    entry = catalogue.get(video_filename)
    crop_x_offset = POSITION_TO_OFFSET.get(entry.speaker_position, 0.5)

    portrait_config = PortraitConfig(crop_x_offset=crop_x_offset)
    converter = PortraitConverter()
    renderer = CaptionRenderer()

    total = len(project.clips)
    for i, clip in enumerate(project.clips):
        source_path = clip.raw_clip_path
        if not source_path or not Path(source_path).exists():
            if progress_callback:
                progress_callback((i + 1) / total * 100)
            continue

        # Re-crop to portrait with updated offset
        portrait_path = project.portrait_clips_dir / f"{clip.clip_id}_portrait.mp4"
        if portrait_path.exists():
            portrait_path.unlink()

        converter.convert(
            input_path=source_path,
            output_path=portrait_path,
            config=portrait_config,
        )
        clip.portrait_clip_path = portrait_path

        # Re-burn captions on new portrait clip
        final_path = project.final_clips_dir / f"{clip.clip_id}_final.mp4"
        if final_path.exists():
            final_path.unlink()

        # Build minimal caption track from hook text
        caption_track = CaptionTrack()
        if clip.segment.hook_text:
            caption_track.add_caption(
                text=clip.segment.hook_text,
                start_time=0.0,
                end_time=min(3.0, clip.segment.duration),
            )

        renderer.render(
            input_path=portrait_path,
            output_path=final_path,
            caption_track=caption_track,
        )
        clip.captioned_clip_path = final_path

        if progress_callback:
            progress_callback((i + 1) / total * 100)

    project.save()
