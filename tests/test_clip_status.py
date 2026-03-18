import pytest
from clip_video.modes.highlights import HighlightClip, ClipStatus, ScheduleEntry
from clip_video.llm.base import HighlightSegment


def _make_segment():
    return HighlightSegment(
        start_time=10.0, end_time=40.0,
        summary="Test", hook_text="Hook", reason="Reason",
    )


class TestClipStatus:
    def test_default_status_is_new(self):
        clip = HighlightClip(clip_id="clip_01", segment=_make_segment(), source_video="v.mp4")
        assert clip.status == ClipStatus.NEW

    def test_status_serialization(self):
        clip = HighlightClip(
            clip_id="clip_01", segment=_make_segment(),
            source_video="v.mp4", status=ClipStatus.SELECTED,
        )
        d = clip.to_dict()
        assert d["status"] == "selected"
        restored = HighlightClip.from_dict(d)
        assert restored.status == ClipStatus.SELECTED

    def test_missing_status_defaults_to_new(self):
        d = HighlightClip(
            clip_id="clip_01", segment=_make_segment(), source_video="v.mp4",
        ).to_dict()
        del d["status"]
        restored = HighlightClip.from_dict(d)
        assert restored.status == ClipStatus.NEW


class TestScheduleEntry:
    def test_to_dict_roundtrip(self):
        entry = ScheduleEntry(platform="linkedin", date="2026-03-25")
        assert ScheduleEntry.from_dict(entry.to_dict()) == entry


class TestClipSchedule:
    def test_default_schedule_empty(self):
        clip = HighlightClip(clip_id="clip_01", segment=_make_segment(), source_video="v.mp4")
        assert clip.schedule == []

    def test_schedule_serialization(self):
        clip = HighlightClip(
            clip_id="clip_01", segment=_make_segment(), source_video="v.mp4",
            status=ClipStatus.SCHEDULED,
            schedule=[ScheduleEntry(platform="linkedin", date="2026-03-25")],
        )
        d = clip.to_dict()
        restored = HighlightClip.from_dict(d)
        assert len(restored.schedule) == 1
        assert restored.schedule[0].platform == "linkedin"
        assert restored.schedule[0].date == "2026-03-25"
