from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from clip_video.storage import atomic_write_json, read_json, NotFoundError

VALID_POSITIONS = {"left", "center", "right"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@dataclass
class VideoEntry:
    speaker: str = ""
    speaker_position: str = "left"
    youtube_url: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "speaker_position": self.speaker_position,
            "youtube_url": self.youtube_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VideoEntry:
        position = data.get("speaker_position", "left")
        if position not in VALID_POSITIONS:
            position = "left"
        return cls(
            speaker=data.get("speaker", ""),
            speaker_position=position,
            youtube_url=data.get("youtube_url", ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoEntry):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass
class VideoCatalogue:
    entries: dict[str, VideoEntry] = field(default_factory=dict)

    def get(self, filename: str) -> VideoEntry:
        return self.entries.get(filename, VideoEntry())

    def set(self, filename: str, entry: VideoEntry) -> None:
        self.entries[filename] = entry

    def to_dict(self) -> dict:
        return {name: entry.to_dict() for name, entry in self.entries.items()}

    @classmethod
    def from_dict(cls, data: dict) -> VideoCatalogue:
        entries = {name: VideoEntry.from_dict(entry_data) for name, entry_data in data.items()}
        return cls(entries=entries)


def load_catalogue(path: Path) -> VideoCatalogue:
    try:
        data = read_json(path)
    except NotFoundError:
        return VideoCatalogue()
    if not isinstance(data, dict):
        return VideoCatalogue()
    return VideoCatalogue.from_dict(data)


def save_catalogue(path: Path, catalogue: VideoCatalogue) -> None:
    atomic_write_json(path, catalogue.to_dict())


def scaffold_catalogue(videos_dir: Path, existing: VideoCatalogue | None = None) -> VideoCatalogue:
    catalogue = VideoCatalogue()
    if existing is not None:
        catalogue.entries.update(existing.entries)

    for file_path in videos_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            if file_path.name not in catalogue.entries:
                catalogue.set(file_path.name, VideoEntry())

    return catalogue
