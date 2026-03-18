from __future__ import annotations

import json
import pytest
from pathlib import Path
from clip_video.catalogue import (
    VideoCatalogue, VideoEntry, load_catalogue, save_catalogue, scaffold_catalogue,
)


class TestVideoEntry:
    def test_defaults(self):
        entry = VideoEntry()
        assert entry.speaker == ""
        assert entry.speaker_position == "left"
        assert entry.youtube_url == ""

    def test_from_dict(self):
        entry = VideoEntry.from_dict({
            "speaker": "Jane Smith",
            "speaker_position": "right",
            "youtube_url": "https://youtube.com/watch?v=abc",
        })
        assert entry.speaker == "Jane Smith"
        assert entry.speaker_position == "right"

    def test_to_dict_roundtrip(self):
        entry = VideoEntry(speaker="Bob", speaker_position="center", youtube_url="https://example.com")
        assert VideoEntry.from_dict(entry.to_dict()) == entry

    def test_invalid_position_defaults_to_left(self):
        entry = VideoEntry.from_dict({"speaker_position": "invalid"})
        assert entry.speaker_position == "left"


class TestVideoCatalogue:
    def test_empty(self):
        cat = VideoCatalogue()
        assert cat.entries == {}

    def test_get_entry_missing_returns_default(self):
        cat = VideoCatalogue()
        entry = cat.get("nonexistent.mp4")
        assert entry.speaker == ""
        assert entry.speaker_position == "left"

    def test_set_and_get(self):
        cat = VideoCatalogue()
        cat.set("video.mp4", VideoEntry(speaker="Alice"))
        assert cat.get("video.mp4").speaker == "Alice"


class TestCatalogueIO:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "videos.json"
        cat = VideoCatalogue()
        cat.set("talk.mp4", VideoEntry(speaker="Jane", speaker_position="right"))
        save_catalogue(path, cat)
        loaded = load_catalogue(path)
        assert loaded.get("talk.mp4").speaker == "Jane"

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "videos.json"
        cat = load_catalogue(path)
        assert cat.entries == {}


class TestScaffoldCatalogue:
    def test_creates_entries_for_videos(self, tmp_path):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "talk-a.mp4").touch()
        (videos_dir / "talk-b.mp4").touch()
        (videos_dir / "notes.txt").touch()
        cat = scaffold_catalogue(videos_dir)
        assert "talk-a.mp4" in cat.entries
        assert "talk-b.mp4" in cat.entries
        assert "notes.txt" not in cat.entries

    def test_preserves_existing_entries(self, tmp_path):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "talk-a.mp4").touch()
        (videos_dir / "talk-b.mp4").touch()
        existing = VideoCatalogue()
        existing.set("talk-a.mp4", VideoEntry(speaker="Alice", speaker_position="right"))
        cat = scaffold_catalogue(videos_dir, existing=existing)
        assert cat.get("talk-a.mp4").speaker == "Alice"
        assert cat.get("talk-a.mp4").speaker_position == "right"
        assert cat.get("talk-b.mp4").speaker == ""
