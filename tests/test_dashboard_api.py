import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from clip_video.dashboard.server import create_app
from clip_video.catalogue import VideoCatalogue, VideoEntry, save_catalogue
from clip_video.modes.highlights import (
    HighlightsProject, HighlightClip, ClipStatus, ScheduleEntry,
)
from clip_video.llm.base import HighlightSegment


def _setup_brand(tmp_path, brand="testbrand"):
    brand_path = tmp_path / brand
    brand_path.mkdir()

    # Config
    (brand_path / "config.json").write_text(json.dumps({"name": brand}))

    # Videos dir
    videos_dir = brand_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "talk-a.mp4").touch()

    # Catalogue
    cat = VideoCatalogue()
    cat.set("talk-a.mp4", VideoEntry(speaker="Alice", speaker_position="left", youtube_url="https://yt.com/a"))
    save_catalogue(brand_path / "videos.json", cat)

    # Highlights project with clips
    proj_dir = brand_path / "highlights" / "talk-a"
    for sub in ["clips/raw", "clips/portrait", "clips/final", "metadata"]:
        (proj_dir / sub).mkdir(parents=True)

    segment = HighlightSegment(
        start_time=10.0, end_time=40.0,
        summary="Great point about testing", hook_text="Testing matters",
        reason="Insightful", topics=["testing"],
    )
    clip = HighlightClip(
        clip_id="clip_01", segment=segment,
        source_video=str(videos_dir / "talk-a.mp4"),
    )
    project = HighlightsProject(
        name="talk-a", brand_name=brand,
        video_path=videos_dir / "talk-a.mp4",
        clips=[clip],
    )
    project._project_root = proj_dir
    project.save()

    return brand_path


@pytest.fixture
def client(tmp_path):
    _setup_brand(tmp_path)
    app = create_app(brand_name="testbrand", brands_root=tmp_path)
    return TestClient(app)


class TestVideosAPI:
    def test_list_videos(self, client):
        resp = client.get("/api/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert any(v["filename"] == "talk-a.mp4" for v in data)

    def test_video_includes_catalogue_data(self, client):
        resp = client.get("/api/videos")
        video = next(v for v in resp.json() if v["filename"] == "talk-a.mp4")
        assert video["speaker"] == "Alice"
        assert video["speaker_position"] == "left"
        assert video["youtube_url"] == "https://yt.com/a"

    def test_video_includes_clips(self, client):
        resp = client.get("/api/videos")
        video = next(v for v in resp.json() if v["filename"] == "talk-a.mp4")
        assert len(video["clips"]) >= 1
        assert video["clips"][0]["clip_id"] == "clip_01"

    def test_update_catalogue_entry(self, client):
        resp = client.patch("/api/videos/talk-a.mp4", json={
            "speaker_position": "right",
            "youtube_url": "https://yt.com/updated",
        })
        assert resp.status_code == 200
        # Verify persisted
        resp = client.get("/api/videos")
        video = next(v for v in resp.json() if v["filename"] == "talk-a.mp4")
        assert video["speaker_position"] == "right"
        assert video["youtube_url"] == "https://yt.com/updated"


class TestClipsAPI:
    def test_update_clip_status(self, client):
        resp = client.patch("/api/clips/clip_01", json={"status": "selected"})
        assert resp.status_code == 200

    def test_schedule_clip(self, client):
        resp = client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin", "date": "2026-03-25",
        })
        assert resp.status_code == 200
        # Verify
        resp = client.get("/api/videos")
        video = next(v for v in resp.json() if v["filename"] == "talk-a.mp4")
        clip = video["clips"][0]
        assert clip["status"] == "scheduled"
        assert len(clip["schedule"]) == 1

    def test_get_schedule(self, client):
        client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin", "date": "2026-03-25",
        })
        resp = client.get("/api/schedule")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) >= 1
        assert entries[0]["date"] == "2026-03-25"

    def test_delete_schedule_entry(self, client):
        client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin", "date": "2026-03-25",
        })
        resp = client.delete("/api/clips/clip_01/schedule/0")
        assert resp.status_code == 200
        clip = resp.json()
        assert len(clip["schedule"]) == 0
        assert clip["status"] == "selected"

    def test_clip_not_found(self, client):
        resp = client.patch("/api/clips/nonexistent", json={"status": "selected"})
        assert resp.status_code == 404


class TestBrandAPI:
    def test_get_brand(self, client):
        resp = client.get("/api/brand")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "testbrand"
        assert "social_platforms" in data
