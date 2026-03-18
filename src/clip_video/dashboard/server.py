from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clip_video.catalogue import load_catalogue, save_catalogue
from clip_video.dashboard.tasks import TaskQueue, TaskStatus
from clip_video.modes.highlights import ClipStatus, HighlightClip, HighlightsProject, ScheduleEntry

STATIC_DIR = Path(__file__).parent / "static"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


class CatalogueUpdate(BaseModel):
    speaker: str | None = None
    speaker_position: str | None = None
    youtube_url: str | None = None


class ClipStatusUpdate(BaseModel):
    status: str


class ScheduleRequest(BaseModel):
    platform: str
    date: str


def _find_clip(
    brand_path: Path, clip_id: str
) -> tuple[HighlightsProject | None, HighlightClip | None]:
    """Scan all highlight projects and return (project, clip) for the given clip_id."""
    highlights_dir = brand_path / "highlights"
    if not highlights_dir.exists():
        return None, None
    for proj_dir in highlights_dir.iterdir():
        state_file = proj_dir / "project_state.json"
        if not state_file.exists():
            continue
        project = HighlightsProject.load(state_file)
        for clip in project.clips:
            if clip.clip_id == clip_id:
                return project, clip
    return None, None


def _clip_to_dict(clip: HighlightClip, project_name: str, brand_path: Path) -> dict[str, Any]:
    """Convert a HighlightClip to the API response dict."""
    seg = clip.segment
    highlights_prefix = f"highlights/{project_name}/clips"
    return {
        "clip_id": clip.clip_id,
        "project": project_name,
        "status": clip.status.value,
        "schedule": [s.to_dict() for s in clip.schedule],
        "summary": seg.summary,
        "hook_text": seg.hook_text,
        "topics": seg.topics,
        "quality_score": seg.quality_score,
        "duration": seg.duration,
        "start_time": seg.start_time,
        "end_time": seg.end_time,
        "raw_path": f"{highlights_prefix}/raw/{clip.clip_id}.mp4",
        "portrait_path": f"{highlights_prefix}/portrait/{clip.clip_id}_portrait.mp4",
        "final_path": f"{highlights_prefix}/final/{clip.clip_id}_final.mp4",
        "thumbnail_url": f"/api/clips/{clip.clip_id}/thumbnail",
    }


def create_app(brand_name: str, brands_root: Path) -> FastAPI:
    app = FastAPI(title=f"clip-video dashboard - {brand_name}")
    brand_path = brands_root / brand_name
    task_queue = TaskQueue()
    app.state.brand_name = brand_name
    app.state.brands_root = brands_root
    app.state.brand_path = brand_path
    app.state.task_queue = task_queue

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "brand": brand_name}

    @app.get("/api/brand")
    def get_brand() -> dict[str, Any]:
        config_file = brand_path / "config.json"
        config_data: dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, encoding="utf-8") as fh:
                config_data = json.load(fh)

        logo_url = None
        logo_dir = brand_path / "logo"
        if logo_dir.exists():
            for logo_file in logo_dir.iterdir():
                if logo_file.is_file() and logo_file.suffix.lower() in IMAGE_EXTENSIONS:
                    logo_url = f"/media/../logo/{logo_file.name}"
                    break

        return {
            "name": config_data.get("name", brand_name),
            "social_platforms": config_data.get("social_platforms", ["linkedin"]),
            "logo_url": logo_url,
        }

    @app.get("/api/videos")
    def list_videos() -> list[dict[str, Any]]:
        catalogue = load_catalogue(brand_path / "videos.json")

        clips_by_video: dict[str, list[dict[str, Any]]] = {}
        highlights_dir = brand_path / "highlights"
        if highlights_dir.exists():
            for proj_dir in highlights_dir.iterdir():
                state_file = proj_dir / "project_state.json"
                if not state_file.exists():
                    continue
                project = HighlightsProject.load(state_file)
                for clip in project.clips:
                    video_filename = Path(clip.source_video).name
                    clips_by_video.setdefault(video_filename, []).append(
                        _clip_to_dict(clip, project.name, brand_path)
                    )

        all_filenames = set(catalogue.entries.keys()) | set(clips_by_video.keys())
        result: list[dict[str, Any]] = []
        for filename in sorted(all_filenames):
            entry = catalogue.get(filename)
            result.append({
                "filename": filename,
                "speaker": entry.speaker,
                "speaker_position": entry.speaker_position,
                "youtube_url": entry.youtube_url,
                "clips": clips_by_video.get(filename, []),
            })
        return result

    @app.patch("/api/videos/{filename}")
    def update_video(filename: str, body: CatalogueUpdate) -> dict[str, Any]:
        cat_path = brand_path / "videos.json"
        catalogue = load_catalogue(cat_path)
        entry = catalogue.get(filename)
        if body.speaker is not None:
            entry.speaker = body.speaker
        if body.speaker_position is not None:
            entry.speaker_position = body.speaker_position
        if body.youtube_url is not None:
            entry.youtube_url = body.youtube_url
        catalogue.set(filename, entry)
        save_catalogue(cat_path, catalogue)
        return entry.to_dict()

    @app.patch("/api/clips/{clip_id}")
    def update_clip(clip_id: str, body: ClipStatusUpdate) -> dict[str, Any]:
        project, clip = _find_clip(brand_path, clip_id)
        if project is None or clip is None:
            raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
        clip.status = ClipStatus(body.status)
        project.save()
        return _clip_to_dict(clip, project.name, brand_path)

    @app.post("/api/clips/{clip_id}/schedule")
    async def schedule_clip(clip_id: str, body: ScheduleRequest) -> dict[str, Any]:
        project, clip = _find_clip(brand_path, clip_id)
        if project is None or clip is None:
            raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
        clip.schedule.append(ScheduleEntry(platform=body.platform, date=body.date))
        if clip.status != ClipStatus.SCHEDULED:
            clip.status = ClipStatus.SCHEDULED

        if body.platform == "youtube" and not clip.landscape_clip_path:
            task_id = task_queue.submit(
                "landscape", target=clip_id,
                description=f"Generating landscape clip for {clip_id}",
            )

            async def run_landscape() -> None:
                from clip_video.dashboard.reprocess import generate_landscape_clip
                try:
                    task_queue.update(task_id, status=TaskStatus.RUNNING)
                    await asyncio.to_thread(generate_landscape_clip, project, clip)
                    project.save()
                    task_queue.update(task_id, status=TaskStatus.COMPLETED, progress=100.0)
                except Exception as e:
                    task_queue.update(task_id, status=TaskStatus.FAILED, error=str(e))

            asyncio.create_task(run_landscape())

        project.save()
        return _clip_to_dict(clip, project.name, brand_path)

    @app.delete("/api/clips/{clip_id}/schedule/{index}")
    def delete_schedule_entry(clip_id: str, index: int) -> dict[str, Any]:
        project, clip = _find_clip(brand_path, clip_id)
        if project is None or clip is None:
            raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")
        if index < 0 or index >= len(clip.schedule):
            raise HTTPException(status_code=404, detail=f"Schedule index {index} out of range")
        clip.schedule.pop(index)
        if not clip.schedule and clip.status == ClipStatus.SCHEDULED:
            clip.status = ClipStatus.SELECTED
        project.save()
        return _clip_to_dict(clip, project.name, brand_path)

    @app.get("/api/schedule")
    def get_schedule() -> list[dict[str, Any]]:
        catalogue = load_catalogue(brand_path / "videos.json")
        entries: list[dict[str, Any]] = []
        highlights_dir = brand_path / "highlights"
        if not highlights_dir.exists():
            return entries
        for proj_dir in highlights_dir.iterdir():
            state_file = proj_dir / "project_state.json"
            if not state_file.exists():
                continue
            project = HighlightsProject.load(state_file)
            for clip in project.clips:
                if not clip.schedule:
                    continue
                video_filename = Path(clip.source_video).name
                cat_entry = catalogue.get(video_filename)
                for sched in clip.schedule:
                    entries.append({
                        "clip_id": clip.clip_id,
                        "project": project.name,
                        "speaker": cat_entry.speaker,
                        "date": sched.date,
                        "platform": sched.platform,
                        "summary": clip.segment.summary,
                        "hook_text": clip.segment.hook_text,
                    })
        return entries

    @app.post("/api/videos/{filename}/reprocess")
    async def reprocess_video(filename: str) -> dict[str, str]:
        highlights_dir = brand_path / "highlights"
        if not highlights_dir.exists():
            raise HTTPException(status_code=404, detail="No highlights found")

        project = None
        for proj_dir in highlights_dir.iterdir():
            state_file = proj_dir / "project_state.json"
            if not state_file.exists():
                continue
            p = HighlightsProject.load(state_file)
            if Path(p.video_path).name == filename:
                project = p
                break

        if project is None:
            raise HTTPException(status_code=404, detail=f"No project found for {filename}")

        task_id = task_queue.submit(
            "reprocess", target=filename,
            description=f"Reprocessing clips for {filename}",
        )

        async def run_reprocess() -> None:
            from clip_video.dashboard.reprocess import reprocess_video_clips
            try:
                task_queue.update(task_id, status=TaskStatus.RUNNING)

                def on_progress(pct: float) -> None:
                    task_queue.update(task_id, progress=pct)

                await asyncio.to_thread(
                    reprocess_video_clips, project, brand_path / "videos.json", on_progress
                )
                task_queue.update(task_id, status=TaskStatus.COMPLETED, progress=100.0)
            except Exception as e:
                task_queue.update(task_id, status=TaskStatus.FAILED, error=str(e))

        asyncio.create_task(run_reprocess())
        return {"task_id": task_id}

    @app.get("/api/clips/{clip_id}/thumbnail")
    def get_thumbnail(clip_id: str) -> FileResponse:
        """Get or generate a thumbnail for a clip."""
        project, clip = _find_clip(brand_path, clip_id)
        if project is None or clip is None:
            raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found")

        thumbnails_dir = project.clips_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumbnails_dir / f"{clip_id}.jpg"

        if not thumb_path.exists():
            # Generate from best available clip version
            source = clip.captioned_clip_path or clip.portrait_clip_path or clip.raw_clip_path
            if not source or not Path(source).exists():
                raise HTTPException(status_code=404, detail="No clip file found")
            try:
                from clip_video.ffmpeg import FFmpegWrapper
                ffmpeg = FFmpegWrapper()
                ffmpeg.get_thumbnail(source, thumb_path, timestamp=1.0, width=320)
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

        return FileResponse(str(thumb_path), media_type="image/jpeg")

    @app.get("/api/tasks")
    def list_tasks() -> list[dict[str, Any]]:
        return [t.to_dict() for t in task_queue.list_tasks()]

    @app.get("/api/tasks/stream")
    async def tasks_stream() -> StreamingResponse:
        async def event_generator():
            while True:
                tasks_data = [t.to_dict() for t in task_queue.list_tasks()]
                yield f"data: {json.dumps(tasks_data)}\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount highlights directory for serving clip videos
    highlights_dir = brand_path / "highlights"
    if highlights_dir.exists():
        app.mount("/media", StaticFiles(directory=str(highlights_dir)), name="media")

    return app
