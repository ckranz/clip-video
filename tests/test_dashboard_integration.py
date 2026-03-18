"""End-to-end integration test for the dashboard workflow."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from clip_video.catalogue import VideoCatalogue, VideoEntry, save_catalogue
from clip_video.dashboard.server import create_app
from clip_video.llm.base import HighlightSegment
from clip_video.modes.highlights import HighlightClip, HighlightsProject


def _setup_full_brand(tmp_path):
    """Create a brand with videos, catalogue, and a highlights project."""
    brand = "kcduk"
    brand_path = tmp_path / brand
    brand_path.mkdir()

    # Brand config with social platforms
    (brand_path / "config.json").write_text(json.dumps({
        "name": "KCD UK",
        "social_platforms": ["linkedin", "youtube"],
    }))

    # Logo
    logo_dir = brand_path / "logo"
    logo_dir.mkdir()
    (logo_dir / "logo.png").write_bytes(b"PNG fake")

    # Videos
    videos_dir = brand_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "keynote-alice.mp4").touch()
    (videos_dir / "lightning-bob.mp4").touch()

    # Catalogue
    cat = VideoCatalogue()
    cat.set("keynote-alice.mp4", VideoEntry(
        speaker="Alice Smith",
        speaker_position="left",
        youtube_url="https://youtube.com/watch?v=alice",
    ))
    cat.set("lightning-bob.mp4", VideoEntry(
        speaker="Bob Jones",
        speaker_position="center",
        youtube_url="https://youtube.com/watch?v=bob",
    ))
    save_catalogue(brand_path / "videos.json", cat)

    # Highlights projects
    for proj_name, video_name, clips_data in [
        ("keynote-alice", "keynote-alice.mp4", [
            ("clip_01", "Great opening keynote", "Cloud native is the future", 10.0, 45.0, 0.9),
            ("clip_02", "Kubernetes deep dive", "Let me show you something cool", 120.0, 160.0, 0.75),
        ]),
        ("lightning-bob", "lightning-bob.mp4", [
            ("clip_03", "Quick demo of tooling", "This tool changed everything", 5.0, 35.0, 0.85),
        ]),
    ]:
        proj_dir = brand_path / "highlights" / proj_name
        for sub in ["clips/raw", "clips/portrait", "clips/final", "metadata"]:
            (proj_dir / sub).mkdir(parents=True)

        clips = []
        for clip_id, summary, hook, start, end, score in clips_data:
            # Create fake clip files
            (proj_dir / "clips" / "raw" / f"{clip_id}.mp4").write_bytes(b"raw")
            (proj_dir / "clips" / "portrait" / f"{clip_id}_portrait.mp4").write_bytes(b"portrait")
            (proj_dir / "clips" / "final" / f"{clip_id}_final.mp4").write_bytes(b"final")

            segment = HighlightSegment(
                start_time=start, end_time=end,
                summary=summary, hook_text=hook,
                reason="Good content", topics=["kubernetes"],
                quality_score=score,
            )
            clips.append(HighlightClip(
                clip_id=clip_id, segment=segment,
                source_video=str(videos_dir / video_name),
                raw_clip_path=proj_dir / "clips" / "raw" / f"{clip_id}.mp4",
                portrait_clip_path=proj_dir / "clips" / "portrait" / f"{clip_id}_portrait.mp4",
                captioned_clip_path=proj_dir / "clips" / "final" / f"{clip_id}_final.mp4",
            ))

        project = HighlightsProject(
            name=proj_name, brand_name=brand,
            video_path=videos_dir / video_name,
            clips=clips,
        )
        project._project_root = proj_dir
        project.save()

    return brand, brand_path


class TestDashboardIntegration:
    @pytest.fixture
    def setup(self, tmp_path):
        brand, brand_path = _setup_full_brand(tmp_path)
        app = create_app(brand_name=brand, brands_root=tmp_path)
        client = TestClient(app)
        return client, brand_path

    def test_brand_info(self, setup):
        client, _ = setup
        resp = client.get("/api/brand")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "KCD UK"
        assert "linkedin" in data["social_platforms"]
        assert "youtube" in data["social_platforms"]
        assert data["logo_url"] is not None

    def test_list_all_videos_with_clips(self, setup):
        client, _ = setup
        resp = client.get("/api/videos")
        assert resp.status_code == 200
        videos = resp.json()
        assert len(videos) == 2

        alice_video = next(v for v in videos if v["filename"] == "keynote-alice.mp4")
        assert alice_video["speaker"] == "Alice Smith"
        assert alice_video["speaker_position"] == "left"
        assert len(alice_video["clips"]) == 2
        assert alice_video["clips"][0]["clip_id"] == "clip_01"

        bob_video = next(v for v in videos if v["filename"] == "lightning-bob.mp4")
        assert len(bob_video["clips"]) == 1

    def test_update_speaker_position(self, setup):
        client, _ = setup
        resp = client.patch("/api/videos/keynote-alice.mp4", json={
            "speaker_position": "right",
        })
        assert resp.status_code == 200
        assert resp.json()["speaker_position"] == "right"

        # Verify persisted
        resp = client.get("/api/videos")
        alice = next(v for v in resp.json() if v["filename"] == "keynote-alice.mp4")
        assert alice["speaker_position"] == "right"

    def test_select_and_skip_clips(self, setup):
        client, _ = setup

        # Select clip_01
        resp = client.patch("/api/clips/clip_01", json={"status": "selected"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "selected"

        # Skip clip_02
        resp = client.patch("/api/clips/clip_02", json={"status": "skipped"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

        # Verify in video listing
        resp = client.get("/api/videos")
        alice = next(v for v in resp.json() if v["filename"] == "keynote-alice.mp4")
        statuses = {c["clip_id"]: c["status"] for c in alice["clips"]}
        assert statuses["clip_01"] == "selected"
        assert statuses["clip_02"] == "skipped"

    def test_schedule_clip_and_view_calendar(self, setup):
        client, _ = setup

        # Schedule clip_01 for LinkedIn
        resp = client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin",
            "date": "2026-03-25",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "scheduled"

        # Schedule same clip for YouTube on different date
        resp = client.post("/api/clips/clip_01/schedule", json={
            "platform": "youtube",
            "date": "2026-03-26",
        })
        assert resp.status_code == 200
        assert len(resp.json()["schedule"]) == 2

        # Check schedule view
        resp = client.get("/api/schedule")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 2
        platforms = {e["platform"] for e in entries}
        assert platforms == {"linkedin", "youtube"}

    def test_remove_schedule_entry(self, setup):
        client, _ = setup

        # Schedule then remove
        client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin", "date": "2026-03-25",
        })
        resp = client.delete("/api/clips/clip_01/schedule/0")
        assert resp.status_code == 200
        assert len(resp.json()["schedule"]) == 0
        # Status reverts to selected since no schedules remain
        assert resp.json()["status"] == "selected"

    def test_reprocess_creates_task(self, setup):
        client, _ = setup

        resp = client.post("/api/videos/keynote-alice.mp4/reprocess")
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        assert task_id is not None

        # Verify task appears in task list
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert any(t["task_id"] == task_id for t in tasks)

    def test_full_workflow(self, setup):
        """End-to-end: review clips, select best, schedule, verify calendar."""
        client, _ = setup

        # 1. Review videos
        resp = client.get("/api/videos")
        videos = resp.json()
        assert len(videos) == 2

        # 2. Select the best clip from Alice's talk
        client.patch("/api/clips/clip_01", json={"status": "selected"})
        client.patch("/api/clips/clip_02", json={"status": "skipped"})

        # 3. Select Bob's clip too
        client.patch("/api/clips/clip_03", json={"status": "selected"})

        # 4. Schedule Alice for Monday, Bob for Wednesday
        client.post("/api/clips/clip_01/schedule", json={
            "platform": "linkedin", "date": "2026-03-23",
        })
        client.post("/api/clips/clip_03/schedule", json={
            "platform": "linkedin", "date": "2026-03-25",
        })

        # 5. Verify schedule
        resp = client.get("/api/schedule")
        entries = resp.json()
        assert len(entries) == 2
        dates = sorted(e["date"] for e in entries)
        assert dates == ["2026-03-23", "2026-03-25"]

        # 6. Update Alice's speaker position and trigger reprocess
        client.patch("/api/videos/keynote-alice.mp4", json={"speaker_position": "right"})
        resp = client.post("/api/videos/keynote-alice.mp4/reprocess")
        assert resp.status_code == 200
