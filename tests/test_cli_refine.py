"""Tests for the CLI refine command."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

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


def _setup_brand(tmp_path: Path, brand_name: str = "testbrand", transcripts: dict[str, TranscriptionResult] | None = None) -> Path:
    """Set up a brand directory with config and optional transcripts."""
    brand_path = tmp_path / "brands" / brand_name
    brand_path.mkdir(parents=True)
    transcripts_dir = brand_path / "transcripts"
    transcripts_dir.mkdir()

    config = {
        "name": brand_name,
        "vocabulary": {},
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
    }
    (brand_path / "config.json").write_text(json.dumps(config))

    if transcripts:
        for filename, result in transcripts.items():
            result.save(transcripts_dir / filename)

    return brand_path


class TestRefineCommandBasic:
    """Tests for basic refine command behavior."""

    def test_brand_not_found(self):
        """Refine exits with error when brand doesn't exist."""
        with patch("clip_video.cli.brand_exists", return_value=False):
            result = runner.invoke(app, ["refine", "nonexistent"])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_no_transcripts(self, tmp_path):
        """Refine handles empty transcripts directory gracefully."""
        _setup_brand(tmp_path, "emptybrand")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=tmp_path / "brands" / "emptybrand"),
            patch("clip_video.cli.load_brand_config") as mock_config,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["refine", "emptybrand"])

        assert result.exit_code == 0
        assert "No transcript files found" in result.output


class TestRefineCommandFileFiltering:
    """Tests for file inclusion/exclusion logic."""

    def test_excludes_dot_files(self, tmp_path):
        """Dot files (progress tracker) should be excluded."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })
        # Add a dot file
        (brand_path / "transcripts" / ".progress.json").write_text("{}")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            # Return unchanged segments (no corrections)
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[]),
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        # Should process talk1.json but not .progress.json
        assert "talk1.json" in result.output

    def test_excludes_backup_files(self, tmp_path):
        """Backup files (.pre-refine.json, .refine-*.json) should be excluded."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })
        transcripts_dir = brand_path / "transcripts"
        # Add backup files
        _make_transcript().save(transcripts_dir / "talk1.pre-refine.json")
        _make_transcript().save(transcripts_dir / "talk1.refine-2025-01-01T120000.json")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[]),
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        # Should only process talk1.json, not the backup files
        assert mock_refine.call_count == 1

    def test_video_filter(self, tmp_path):
        """--video option filters by substring match on filename."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
            "talk2.json": _make_transcript("talk2.mp4"),
            "keynote.json": _make_transcript("keynote.mp4"),
        })

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[MagicMock()]),  # 1 correction so file is refined
            )
            result = runner.invoke(app, ["refine", "mybrand", "--video", "keynote"])

        assert result.exit_code == 0
        assert "keynote.json" in result.output
        assert mock_refine.call_count == 1
        assert "Refined: 1" in result.output


class TestRefineCommandBackups:
    """Tests for backup file creation."""

    def test_first_run_creates_pre_refine_backup(self, tmp_path):
        """First refinement should create .pre-refine.json backup."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })
        transcripts_dir = brand_path / "transcripts"

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[MagicMock()]),  # 1 correction
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        backup = transcripts_dir / "talk1.pre-refine.json"
        assert backup.exists(), f"Expected {backup} to exist"

    def test_subsequent_run_creates_timestamped_backup(self, tmp_path):
        """Subsequent refinement should create timestamped backup, not overwrite .pre-refine."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })
        transcripts_dir = brand_path / "transcripts"
        # Simulate previous refinement by creating .pre-refine backup
        _make_transcript().save(transcripts_dir / "talk1.pre-refine.json")

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[MagicMock()]),
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        # .pre-refine should still exist (not overwritten)
        assert (transcripts_dir / "talk1.pre-refine.json").exists()
        # Timestamped backup should be created
        timestamped_backups = list(transcripts_dir.glob("talk1.refine-*.json"))
        assert len(timestamped_backups) == 1, f"Expected 1 timestamped backup, found {timestamped_backups}"


class TestRefineCommandErrorHandling:
    """Tests for error handling during refinement."""

    def test_continues_on_individual_failure(self, tmp_path):
        """Should continue processing other files when one fails."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
            "talk2.json": _make_transcript("talk2.mp4"),
        })

        call_count = 0

        def mock_refine_side_effect(segments, context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM connection failed")
            return (
                _make_transcript().segments,
                MagicMock(corrections=[MagicMock()]),
            )

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.side_effect = mock_refine_side_effect
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        # Should report both success and failure
        assert "Error" in result.output or "error" in result.output
        assert "1" in result.output  # At least 1 refined

    def test_provider_not_available(self, tmp_path):
        """Should exit with error when LLM provider is not available."""
        _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=tmp_path / "brands" / "mybrand"),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=False),
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 1
        assert "not available" in result.output


class TestRefineCommandOptions:
    """Tests for CLI option handling."""

    def test_provider_and_model_options(self, tmp_path):
        """--provider and --model should override brand config."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner") as MockRefiner,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="claude",
                llm_model=None,
            )
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = True
            mock_instance.refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[]),
            )
            MockRefiner.return_value = mock_instance

            result = runner.invoke(app, [
                "refine", "mybrand",
                "--provider", "ollama",
                "--model", "mistral",
            ])

        assert result.exit_code == 0
        # Verify LLMConfig was created with overridden values
        call_args = MockRefiner.call_args
        llm_config = call_args[0][0]
        assert llm_config.provider.value == "ollama"
        assert llm_config.model == "mistral"

    def test_talk_title_and_description(self, tmp_path):
        """--talk-title and --talk-description should be passed to RefinementContext."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner") as MockRefiner,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={"kubernetes": ["cooper netties"]},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = True
            mock_instance.refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[]),
            )
            MockRefiner.return_value = mock_instance

            result = runner.invoke(app, [
                "refine", "mybrand",
                "--talk-title", "Kubernetes 101",
                "--talk-description", "Intro to K8s",
            ])

        assert result.exit_code == 0
        # Verify refine was called with context
        refine_call = mock_instance.refine.call_args
        context = refine_call[1].get("context") or refine_call[0][1] if len(refine_call[0]) > 1 else refine_call[1].get("context")
        assert context.talk_title == "Kubernetes 101"
        assert context.talk_description == "Intro to K8s"


class TestRefineCommandSummary:
    """Tests for summary output."""

    def test_summary_panel_displayed(self, tmp_path):
        """Should display a summary panel at the end."""
        brand_path = _setup_brand(tmp_path, "mybrand", {
            "talk1.json": _make_transcript("talk1.mp4"),
        })

        with (
            patch("clip_video.cli.brand_exists", return_value=True),
            patch("clip_video.cli.get_brand_path", return_value=brand_path),
            patch("clip_video.cli.load_brand_config") as mock_config,
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.is_available", return_value=True),
            patch("clip_video.transcription.llm_refine.TranscriptRefiner.refine") as mock_refine,
        ):
            mock_config.return_value = MagicMock(
                vocabulary={},
                llm_provider="ollama",
                llm_model="llama3.2",
            )
            mock_refine.return_value = (
                _make_transcript().segments,
                MagicMock(corrections=[MagicMock(), MagicMock()]),  # 2 corrections
            )
            result = runner.invoke(app, ["refine", "mybrand"])

        assert result.exit_code == 0
        # Summary should include refinement stats
        assert "Refined" in result.output or "refined" in result.output
