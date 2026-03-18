from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from clip_video.dashboard.reprocess import POSITION_TO_OFFSET, reprocess_video_clips
from clip_video.catalogue import VideoCatalogue, VideoEntry, save_catalogue
from clip_video.modes.highlights import HighlightsProject, HighlightClip
from clip_video.llm.base import HighlightSegment


class TestPositionMapping:
    def test_left_offset(self):
        assert POSITION_TO_OFFSET["left"] == 0.25

    def test_center_offset(self):
        assert POSITION_TO_OFFSET["center"] == 0.5

    def test_right_offset(self):
        assert POSITION_TO_OFFSET["right"] == 0.75

    def test_unknown_position_defaults_center(self):
        assert POSITION_TO_OFFSET.get("unknown", 0.5) == 0.5


class TestReprocessPipeline:
    def _make_project(self, tmp_path):
        """Create a minimal project with one clip for testing."""
        brand_path = tmp_path / "brand"
        brand_path.mkdir()

        proj_dir = brand_path / "highlights" / "talk-a"
        for sub in ["clips/raw", "clips/portrait", "clips/final", "metadata"]:
            (proj_dir / sub).mkdir(parents=True)

        # Create a fake raw clip file
        raw_clip = proj_dir / "clips" / "raw" / "clip_01.mp4"
        raw_clip.write_bytes(b"fake video data")

        # Create fake source video
        video_path = brand_path / "videos" / "talk-a.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake source")

        segment = HighlightSegment(
            start_time=10.0, end_time=40.0,
            summary="Great point", hook_text="Testing matters",
            reason="Insightful",
        )
        clip = HighlightClip(
            clip_id="clip_01", segment=segment,
            source_video=str(video_path),
            raw_clip_path=raw_clip,
        )
        project = HighlightsProject(
            name="talk-a", brand_name="brand",
            video_path=video_path,
            clips=[clip],
        )
        project._project_root = proj_dir
        return project, brand_path

    @patch("clip_video.dashboard.reprocess.CaptionRenderer")
    @patch("clip_video.dashboard.reprocess.PortraitConverter")
    def test_reprocess_calls_converter_with_correct_offset(
        self, MockConverter, MockRenderer, tmp_path
    ):
        project, brand_path = self._make_project(tmp_path)

        # Set up catalogue with right position
        cat = VideoCatalogue()
        cat.set("talk-a.mp4", VideoEntry(speaker_position="right"))
        save_catalogue(brand_path / "videos.json", cat)

        mock_converter = MockConverter.return_value
        mock_converter.convert.return_value = Path("output.mp4")
        mock_renderer = MockRenderer.return_value
        mock_renderer.render.return_value = Path("final.mp4")

        reprocess_video_clips(project, brand_path / "videos.json")

        # Check converter was called
        mock_converter.convert.assert_called_once()
        call_kwargs = mock_converter.convert.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.crop_x_offset == 0.75  # right position

    @patch("clip_video.dashboard.reprocess.CaptionRenderer")
    @patch("clip_video.dashboard.reprocess.PortraitConverter")
    def test_progress_callback_called(self, MockConverter, MockRenderer, tmp_path):
        project, brand_path = self._make_project(tmp_path)
        cat = VideoCatalogue()
        cat.set("talk-a.mp4", VideoEntry(speaker_position="center"))
        save_catalogue(brand_path / "videos.json", cat)

        mock_converter = MockConverter.return_value
        mock_converter.convert.return_value = Path("output.mp4")
        mock_renderer = MockRenderer.return_value
        mock_renderer.render.return_value = Path("final.mp4")

        progress_values = []
        reprocess_video_clips(
            project, brand_path / "videos.json",
            progress_callback=lambda p: progress_values.append(p),
        )

        assert len(progress_values) == 1
        assert progress_values[0] == 100.0

    @patch("clip_video.dashboard.reprocess.CaptionRenderer")
    @patch("clip_video.dashboard.reprocess.PortraitConverter")
    def test_project_saved_after_reprocess(self, MockConverter, MockRenderer, tmp_path):
        project, brand_path = self._make_project(tmp_path)
        cat = VideoCatalogue()
        cat.set("talk-a.mp4", VideoEntry(speaker_position="left"))
        save_catalogue(brand_path / "videos.json", cat)

        mock_converter = MockConverter.return_value
        mock_converter.convert.return_value = Path("output.mp4")
        mock_renderer = MockRenderer.return_value
        mock_renderer.render.return_value = Path("final.mp4")

        reprocess_video_clips(project, brand_path / "videos.json")

        # Project state should be saved (state file should exist)
        assert project.state_file.exists()
