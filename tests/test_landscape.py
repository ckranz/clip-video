import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from clip_video.modes.highlights import HighlightClip, HighlightsProject, ClipStatus
from clip_video.llm.base import HighlightSegment


def _make_segment(**kwargs):
    defaults = dict(
        start_time=10.0, end_time=40.0,
        summary="Test", hook_text="Hook", reason="R",
    )
    defaults.update(kwargs)
    return HighlightSegment(**defaults)


class TestLandscapeClipPath:
    def test_default_landscape_path_is_none(self):
        clip = HighlightClip(clip_id="c1", segment=_make_segment(), source_video="v.mp4")
        assert clip.landscape_clip_path is None

    def test_landscape_path_serialization(self):
        clip = HighlightClip(
            clip_id="c1", segment=_make_segment(), source_video="v.mp4",
            landscape_clip_path=Path("/tmp/landscape.mp4"),
        )
        d = clip.to_dict()
        assert d["landscape_clip_path"] == str(Path("/tmp/landscape.mp4"))
        restored = HighlightClip.from_dict(d)
        assert restored.landscape_clip_path == Path("/tmp/landscape.mp4")

    def test_missing_landscape_path_backwards_compatible(self):
        d = HighlightClip(clip_id="c1", segment=_make_segment(), source_video="v.mp4").to_dict()
        del d["landscape_clip_path"]
        restored = HighlightClip.from_dict(d)
        assert restored.landscape_clip_path is None


class TestGenerateLandscapeClip:
    @patch("clip_video.dashboard.reprocess.CaptionRenderer")
    def test_generates_landscape_from_raw(self, MockRenderer, tmp_path):
        from clip_video.dashboard.reprocess import generate_landscape_clip

        proj_dir = tmp_path / "project"
        for sub in ["clips/raw", "clips/landscape"]:
            (proj_dir / sub).mkdir(parents=True)

        raw_clip = proj_dir / "clips" / "raw" / "c1.mp4"
        raw_clip.write_bytes(b"fake")

        clip = HighlightClip(
            clip_id="c1", segment=_make_segment(),
            source_video="v.mp4", raw_clip_path=raw_clip,
        )
        project = HighlightsProject(
            name="proj", brand_name="brand", video_path=tmp_path / "v.mp4",
            clips=[clip],
        )
        project._project_root = proj_dir

        mock_renderer = MockRenderer.return_value
        mock_renderer.render.return_value = Path("output.mp4")

        result = generate_landscape_clip(project, clip)
        mock_renderer.render.assert_called_once()
        assert clip.landscape_clip_path is not None
        assert "landscape" in str(result)

    def test_raises_when_raw_clip_missing(self, tmp_path):
        from clip_video.dashboard.reprocess import generate_landscape_clip

        proj_dir = tmp_path / "project"
        proj_dir.mkdir()

        clip = HighlightClip(
            clip_id="c1", segment=_make_segment(),
            source_video="v.mp4", raw_clip_path=None,
        )
        project = HighlightsProject(
            name="proj", brand_name="brand", video_path=tmp_path / "v.mp4",
            clips=[clip],
        )
        project._project_root = proj_dir

        with pytest.raises(ValueError, match="Raw clip not found"):
            generate_landscape_clip(project, clip)

    @patch("clip_video.dashboard.reprocess.CaptionRenderer")
    def test_skips_if_output_exists(self, MockRenderer, tmp_path):
        from clip_video.dashboard.reprocess import generate_landscape_clip

        proj_dir = tmp_path / "project"
        (proj_dir / "clips" / "landscape").mkdir(parents=True)
        (proj_dir / "clips" / "raw").mkdir(parents=True)

        raw_clip = proj_dir / "clips" / "raw" / "c1.mp4"
        raw_clip.write_bytes(b"fake")

        existing = proj_dir / "clips" / "landscape" / "c1_landscape.mp4"
        existing.write_bytes(b"already done")

        clip = HighlightClip(
            clip_id="c1", segment=_make_segment(),
            source_video="v.mp4", raw_clip_path=raw_clip,
        )
        project = HighlightsProject(
            name="proj", brand_name="brand", video_path=tmp_path / "v.mp4",
            clips=[clip],
        )
        project._project_root = proj_dir

        result = generate_landscape_clip(project, clip)
        MockRenderer.return_value.render.assert_not_called()
        assert result == existing
