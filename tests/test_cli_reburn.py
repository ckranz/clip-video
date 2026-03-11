"""Tests for the CLI re-burn-captions command."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
from typer.testing import CliRunner

from clip_video.cli import app
from clip_video.transcription.base import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)

runner = CliRunner()


def _make_transcript(video_name: str = "talk1.mp4") -> TranscriptionResult:
    """Create a minimal TranscriptionResult for testing."""
    return TranscriptionResult(
        video_path=video_name,
        text="Hello world",
        segments=[
            TranscriptionSegment(
                text="Hello world",
                start=0.0,
                end=2.0,
                words=[
                    TranscriptionWord(word="Hello", start=0.0, end=1.0, confidence=0.9),
                    TranscriptionWord(word="world", start=1.0, end=2.0, confidence=0.9),
                ],
                confidence=0.9,
            )
        ],
        language="en",
        duration=2.0,
        provider="whisper_local",
        model="medium",
    )


def _make_segment_dict(start: float, end: float, summary: str = "A test highlight") -> dict:
    """Create a valid HighlightSegment dict."""
    return {
        "start_time": start,
        "end_time": end,
        "summary": summary,
        "hook_text": "Watch this",
        "reason": "Interesting content",
        "topics": ["testing"],
        "quality_score": 0.9,
    }


def _make_project_state(video_path: str = "videos/talk1.mp4", clips: list | None = None) -> dict:
    """Create a minimal project_state.json dict."""
    if clips is None:
        clips = [
            {
                "clip_id": "clip_001",
                "segment": _make_segment_dict(10.0, 45.0, "A test highlight"),
                "source_video": video_path,
                "raw_clip_path": "clips/raw/clip_001.mp4",
                "portrait_clip_path": "clips/portrait/clip_001.mp4",
                "captioned_clip_path": "clips/final/clip_001.mp4",
                "metadata": {},
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "clip_id": "clip_002",
                "segment": _make_segment_dict(60.0, 90.0, "Another test"),
                "source_video": video_path,
                "raw_clip_path": "clips/raw/clip_002.mp4",
                "portrait_clip_path": "clips/portrait/clip_002.mp4",
                "captioned_clip_path": "clips/final/clip_002.mp4",
                "metadata": {},
                "created_at": "2026-01-01T00:00:00",
            },
        ]

    return {
        "name": "talk1-highlights",
        "brand_name": "testbrand",
        "video_path": video_path,
        "description_path": None,
        "transcript_text": "Hello world",
        "analysis": {
            "video_id": "talk1",
            "segments": [c["segment"] for c in clips],
            "session_summary": "Test analysis",
        },
        "clips": clips,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


def _setup_brand_with_project(
    tmp_path: Path,
    brand_name: str = "testbrand",
    project_name: str = "talk1-highlights",
    video_name: str = "talk1.mp4",
    create_final_clips: bool = True,
) -> Path:
    """Set up a brand with a highlight project and transcript."""
    brand_path = tmp_path / "brands" / brand_name
    brand_path.mkdir(parents=True)

    # Brand config
    config = {
        "name": brand_name,
        "vocabulary": {},
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
    }
    (brand_path / "config.json").write_text(json.dumps(config))

    # Transcript
    transcripts_dir = brand_path / "transcripts"
    transcripts_dir.mkdir()
    _make_transcript(video_name).save(transcripts_dir / f"{Path(video_name).stem}.json")

    # Highlight project
    project_dir = brand_path / "highlights" / project_name
    project_dir.mkdir(parents=True)

    video_path = f"videos/{video_name}"
    state = _make_project_state(video_path=video_path)
    state["name"] = project_name
    state["brand_name"] = brand_name

    state_file = project_dir / "project_state.json"
    state_file.write_text(json.dumps(state))

    # Create fake final clip files
    if create_final_clips:
        final_dir = project_dir / "clips" / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "clip_001.mp4").write_text("fake video 1")
        (final_dir / "clip_002.mp4").write_text("fake video 2")

    return brand_path


class TestReburnCommandBasic:
    """Tests for basic re-burn-captions command behavior."""

    def test_brand_not_found(self):
        """Exits with error when brand doesn't exist."""
        with patch("clip_video.cli.brand_exists", return_value=False):
            result = runner.invoke(app, ["re-burn-captions", "nonexistent"])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_no_highlight_projects(self, tmp_path):
        """Handles brand with no highlight projects gracefully."""
        brand_path = tmp_path / "brands" / "emptybrand"
        brand_path.mkdir(parents=True)
        (brand_path / "config.json").write_text(json.dumps({
            "name": "emptybrand",
            "vocabulary": {},
            "llm_provider": "ollama",
            "llm_model": "llama3.2",
        }))

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["re-burn-captions", "emptybrand"])

        assert result.exit_code == 0
        assert "No highlight projects found" in result.output


class TestReburnBackups:
    """Tests for backup creation logic."""

    def test_first_run_creates_pre_reburn_backup(self, tmp_path):
        """First run creates pre-reburn/ backup directory with clip files."""
        brand_path = _setup_brand_with_project(tmp_path)

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        backup_dir = brand_path / "highlights" / "talk1-highlights" / "clips" / "backups" / "pre-reburn"
        assert backup_dir.exists()
        assert (backup_dir / "clip_001.mp4").exists()
        assert (backup_dir / "clip_002.mp4").exists()

    def test_subsequent_run_creates_timestamped_backup(self, tmp_path):
        """Subsequent runs create timestamped backup directories."""
        brand_path = _setup_brand_with_project(tmp_path)

        # Simulate first run already happened
        pre_reburn_dir = brand_path / "highlights" / "talk1-highlights" / "clips" / "backups" / "pre-reburn"
        pre_reburn_dir.mkdir(parents=True)
        (pre_reburn_dir / "clip_001.mp4").write_text("original backup")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        # pre-reburn should still exist untouched
        assert pre_reburn_dir.exists()
        assert (pre_reburn_dir / "clip_001.mp4").read_text() == "original backup"

        # Timestamped backup should be created
        backups_dir = brand_path / "highlights" / "talk1-highlights" / "clips" / "backups"
        timestamped = [d for d in backups_dir.iterdir() if d.name.startswith("reburn-")]
        assert len(timestamped) == 1
        assert (timestamped[0] / "clip_001.mp4").exists()
        assert (timestamped[0] / "clip_002.mp4").exists()


class TestReburnFiltering:
    """Tests for --project substring filtering."""

    def test_project_filter_matches(self, tmp_path):
        """--project filters to projects whose name contains the substring."""
        brand_path = _setup_brand_with_project(tmp_path, project_name="talk1-highlights")
        # Add a second project
        project2_dir = brand_path / "highlights" / "keynote-highlights"
        project2_dir.mkdir(parents=True)
        state2 = _make_project_state(video_path="videos/keynote.mp4")
        state2["name"] = "keynote-highlights"
        (project2_dir / "project_state.json").write_text(json.dumps(state2))
        final2 = project2_dir / "clips" / "final"
        final2.mkdir(parents=True)
        (final2 / "clip_001.mp4").write_text("fake")
        # Transcript for keynote
        _make_transcript("keynote.mp4").save(brand_path / "transcripts" / "keynote.json")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand", "--project", "keynote"])

        assert result.exit_code == 0
        assert "keynote-highlights" in result.output
        # burn_captions should only be called once (for keynote, not talk1)
        assert mock_burn.call_count == 1

    def test_project_filter_no_match(self, tmp_path):
        """--project with no match shows appropriate message."""
        brand_path = _setup_brand_with_project(tmp_path)

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["re-burn-captions", "testbrand", "--project", "nonexistent"])

        assert result.exit_code == 0
        assert "No highlight projects found" in result.output

    def test_project_filter_case_insensitive(self, tmp_path):
        """--project filter is case-insensitive."""
        brand_path = _setup_brand_with_project(tmp_path, project_name="TalkOne-highlights")
        # Need transcript for TalkOne
        _make_transcript("TalkOne.mp4").save(brand_path / "transcripts" / "TalkOne.json")
        # Update the project state to use the right video path
        state_file = brand_path / "highlights" / "TalkOne-highlights" / "project_state.json"
        state = json.loads(state_file.read_text())
        state["video_path"] = "videos/TalkOne.mp4"
        state["name"] = "TalkOne-highlights"
        state_file.write_text(json.dumps(state))

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand", "--project", "talkone"])

        assert result.exit_code == 0
        assert mock_burn.call_count == 1


class TestReburnCaptionClearing:
    """Tests for caption path clearing before burn."""

    def test_captioned_clip_path_cleared_before_burn(self, tmp_path):
        """captioned_clip_path must be None on clips passed to burn_captions."""
        brand_path = _setup_brand_with_project(tmp_path)
        captured_projects = []

        def capture_burn(project, transcript_segments=None):
            # Capture the state of clips at the time burn_captions is called
            clip_paths = [c.captioned_clip_path for c in project.clips]
            captured_projects.append(clip_paths)
            return project.clips

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions", side_effect=capture_burn) as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        assert len(captured_projects) == 1
        # All captioned_clip_path values should be None when burn_captions is called
        assert all(p is None for p in captured_projects[0])

    def test_project_state_saved_after_burn(self, tmp_path):
        """Project state is saved after a successful burn."""
        brand_path = _setup_brand_with_project(tmp_path)
        state_file = brand_path / "highlights" / "talk1-highlights" / "project_state.json"
        original_state = json.loads(state_file.read_text())
        original_updated_at = original_state["updated_at"]

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        saved_state = json.loads(state_file.read_text())
        assert saved_state["updated_at"] != original_updated_at
        # captioned_clip_path should be None since mock doesn't set them back
        for clip in saved_state["clips"]:
            assert clip["captioned_clip_path"] is None


class TestReburnErrorHandling:
    """Tests for error handling during re-burn."""

    def test_skips_project_when_transcript_not_found(self, tmp_path):
        """Warns and continues when transcript file is missing."""
        brand_path = _setup_brand_with_project(tmp_path, project_name="talk1-highlights")

        # Remove the transcript file
        transcript_file = brand_path / "transcripts" / "talk1.json"
        transcript_file.unlink()

        # Add a second project with a valid transcript
        project2_dir = brand_path / "highlights" / "talk2-highlights"
        project2_dir.mkdir(parents=True)
        state2 = _make_project_state(video_path="videos/talk2.mp4")
        state2["name"] = "talk2-highlights"
        (project2_dir / "project_state.json").write_text(json.dumps(state2))
        final2 = project2_dir / "clips" / "final"
        final2.mkdir(parents=True)
        (final2 / "clip_001.mp4").write_text("fake")
        (final2 / "clip_002.mp4").write_text("fake")
        _make_transcript("talk2.mp4").save(brand_path / "transcripts" / "talk2.json")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions") as mock_burn,
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        # Should warn about missing transcript
        assert "transcript not found" in result.output.lower() or "Transcript not found" in result.output
        # Should still process the second project
        assert mock_burn.call_count == 1

    def test_skips_project_on_burn_error(self, tmp_path):
        """Continues to next project if burn_captions raises an error."""
        brand_path = _setup_brand_with_project(tmp_path, project_name="talk1-highlights")

        # Add a second project
        project2_dir = brand_path / "highlights" / "talk2-highlights"
        project2_dir.mkdir(parents=True)
        state2 = _make_project_state(video_path="videos/talk2.mp4")
        state2["name"] = "talk2-highlights"
        (project2_dir / "project_state.json").write_text(json.dumps(state2))
        final2 = project2_dir / "clips" / "final"
        final2.mkdir(parents=True)
        (final2 / "clip_001.mp4").write_text("fake")
        (final2 / "clip_002.mp4").write_text("fake")
        _make_transcript("talk2.mp4").save(brand_path / "transcripts" / "talk2.json")

        call_count = 0

        def fail_then_succeed(project, transcript_segments=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("FFmpeg crashed")
            return project.clips

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.modes.highlights.HighlightsProcessor.burn_captions", side_effect=fail_then_succeed),
        ):
            mock_config.return_value = MagicMock(
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["re-burn-captions", "testbrand"])

        assert result.exit_code == 0
        # Should report the error
        assert "error" in result.output.lower() or "Error" in result.output
        # Should have attempted both projects
        assert call_count == 2
