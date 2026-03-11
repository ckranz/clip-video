# Re-burn Captions Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `clip-video re-burn-captions BRAND` command that re-burns captions on existing highlight clips using updated (refined) transcripts.

**Architecture:** New CLI command discovers highlight projects, loads each project's state and its corresponding transcript, backs up existing final clips, clears `captioned_clip_path` so `burn_captions` doesn't skip them, then calls the existing `HighlightsProcessor.burn_captions()` method. No new modules — reuses existing infrastructure entirely.

**Tech Stack:** Typer CLI, HighlightsProcessor, HighlightsProject, TranscriptionResult, Rich console output.

---

### Task 1: Core `re-burn-captions` command

**Files:**
- Modify: `src/clip_video/cli.py` (add new command)
- Create: `tests/test_cli_reburn.py`

**Context:**

Key patterns from the existing codebase:

- `HighlightsProject.load(state_file, config)` loads a project from `project_state.json` (highlights.py:414-420)
- `HighlightsProcessor.burn_captions(project, transcript_segments)` burns captions using transcript segments (highlights.py:833-904). It skips clips where `captioned_clip_path` already exists (line 866-868), so we must clear that field before calling it
- `TranscriptionResult.load(path)` loads transcript with segments (transcription/base.py:147-160)
- `HighlightsProcessor.__init__` requires a `HighlightsConfig` which requires an `LLMConfig` — even though caption burning doesn't use the LLM, the processor constructor initialises one. Use the brand config's llm_provider/llm_model to satisfy this
- `HighlightClip.captioned_clip_path` (highlights.py:86) — set to `None` to force re-burn
- Project state files live at `brands/{brand}/highlights/{project_name}/project_state.json`
- Transcript files live at `brands/{brand}/transcripts/{video_stem}.json`
- The existing `highlights` CLI command (cli.py:1688-1877) shows the pattern for loading transcripts and creating the processor
- The existing `refine` command (cli.py:805+) shows the pattern for batch processing with backup, filtering, and summary output

**Step 1: Write the failing tests**

Create `tests/test_cli_reburn.py`:

```python
"""Tests for the re-burn-captions CLI command."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from typer.testing import CliRunner

from clip_video.cli import app

runner = CliRunner()


def _make_project_state(video_path: str, clips_with_captions: bool = True) -> dict:
    """Create a minimal project state dict for testing."""
    clips = []
    for i in range(2):
        clip = {
            "clip_id": f"clip_{i+1:02d}",
            "segment": {
                "start_time": i * 30.0,
                "end_time": (i + 1) * 30.0,
                "summary": f"Test segment {i+1}",
                "hook_text": f"Hook {i+1}",
                "reason": "Test reason",
                "topics": ["testing"],
                "quality_score": 0.9,
            },
            "source_video": video_path,
            "raw_clip_path": None,
            "portrait_clip_path": None,
            "captioned_clip_path": f"/fake/final/clip_{i+1:02d}_final.mp4" if clips_with_captions else None,
            "metadata": {},
            "created_at": datetime.now().isoformat(),
        }
        clips.append(clip)

    return {
        "name": "test-project",
        "brand_name": "test-brand",
        "video_path": video_path,
        "description_path": None,
        "transcript_text": "[0.0s - 30.0s] Hello world",
        "analysis": {
            "video_id": "test",
            "segments": [c["segment"] for c in clips],
            "summary": "Test analysis",
        },
        "clips": clips,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def _make_transcript(video_name: str) -> dict:
    """Create a minimal transcript dict for testing."""
    return {
        "video_path": f"/videos/{video_name}.mp4",
        "text": "Hello Kubernetes world",
        "segments": [
            {
                "text": "Hello Kubernetes world",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "Kubernetes", "start": 0.5, "end": 1.2},
                    {"word": "world", "start": 1.2, "end": 2.0},
                ],
                "confidence": 1.0,
            }
        ],
        "language": "en",
        "duration": 2.0,
        "provider": "whisper_local",
        "model": "medium",
        "timestamp": datetime.now().isoformat(),
        "vocabulary_corrections": 0,
    }


@pytest.fixture
def brand_dir(tmp_path):
    """Set up a fake brand with highlights project and transcripts."""
    brand_path = tmp_path / "brands" / "test-brand"

    # Create transcripts
    transcripts_dir = brand_path / "transcripts"
    transcripts_dir.mkdir(parents=True)
    transcript = _make_transcript("test-video")
    (transcripts_dir / "test-video.json").write_text(json.dumps(transcript))

    # Create highlights project
    project_dir = brand_path / "highlights" / "test-project"
    final_dir = project_dir / "clips" / "final"
    final_dir.mkdir(parents=True)

    # Create fake final clip files
    (final_dir / "clip_01_final.mp4").write_text("fake video 1")
    (final_dir / "clip_02_final.mp4").write_text("fake video 2")

    # Create project state
    state = _make_project_state(
        str(tmp_path / "videos" / "test-video.mp4"),
        clips_with_captions=True,
    )
    # Point captioned_clip_path to our actual fake files
    state["clips"][0]["captioned_clip_path"] = str(final_dir / "clip_01_final.mp4")
    state["clips"][1]["captioned_clip_path"] = str(final_dir / "clip_02_final.mp4")
    (project_dir / "project_state.json").write_text(json.dumps(state))

    # Create minimal config
    config_path = brand_path / "config.json"
    config_path.write_text(json.dumps({
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
    }))

    return brand_path


class TestReburnCommandBasic:
    """Basic command tests."""

    @patch("clip_video.cli.brand_exists", return_value=False)
    def test_brand_not_found(self, mock_exists):
        result = runner.invoke(app, ["re-burn-captions", "no-such-brand"])
        assert result.exit_code == 1

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_no_projects(self, mock_config, mock_exists, mock_path, tmp_path):
        brand_path = tmp_path / "brands" / "empty-brand"
        highlights_dir = brand_path / "highlights"
        highlights_dir.mkdir(parents=True)

        mock_path.return_value = brand_path
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        result = runner.invoke(app, ["re-burn-captions", "empty-brand"])
        assert result.exit_code == 0
        assert "No highlight projects found" in result.output


class TestReburnBackups:
    """Backup creation tests."""

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_first_run_creates_pre_reburn_backup(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        with patch(
            "clip_video.modes.highlights.HighlightsProcessor.burn_captions"
        ) as mock_burn:
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "test-brand"])

        assert result.exit_code == 0
        backup_dir = brand_dir / "highlights" / "test-project" / "clips" / "backups" / "pre-reburn"
        assert backup_dir.exists()
        assert (backup_dir / "clip_01_final.mp4").exists()
        assert (backup_dir / "clip_02_final.mp4").exists()

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_subsequent_run_creates_timestamped_backup(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        # Create pre-existing backup to simulate previous run
        backup_dir = brand_dir / "highlights" / "test-project" / "clips" / "backups" / "pre-reburn"
        backup_dir.mkdir(parents=True)
        (backup_dir / "clip_01_final.mp4").write_text("old backup")

        with patch(
            "clip_video.modes.highlights.HighlightsProcessor.burn_captions"
        ) as mock_burn:
            mock_burn.return_value = []
            result = runner.invoke(app, ["re-burn-captions", "test-brand"])

        assert result.exit_code == 0
        backups_root = brand_dir / "highlights" / "test-project" / "clips" / "backups"
        timestamped = [d for d in backups_root.iterdir() if d.name.startswith("reburn-")]
        assert len(timestamped) == 1


class TestReburnFiltering:
    """Project filtering tests."""

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_project_filter(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        with patch(
            "clip_video.modes.highlights.HighlightsProcessor.burn_captions"
        ) as mock_burn:
            mock_burn.return_value = []
            result = runner.invoke(app, [
                "re-burn-captions", "test-brand", "--project", "test-project"
            ])

        assert result.exit_code == 0
        assert mock_burn.call_count == 1

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_project_filter_no_match(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        result = runner.invoke(app, [
            "re-burn-captions", "test-brand", "--project", "nonexistent"
        ])
        assert result.exit_code == 0
        assert "No highlight projects found" in result.output


class TestReburnCaptionClearing:
    """Tests that captioned_clip_path is cleared before re-burning."""

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_clears_captioned_clip_path(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        with patch(
            "clip_video.modes.highlights.HighlightsProcessor.burn_captions"
        ) as mock_burn:
            mock_burn.return_value = []

            result = runner.invoke(app, ["re-burn-captions", "test-brand"])

        assert result.exit_code == 0
        # Verify burn_captions was called (meaning clips weren't skipped)
        assert mock_burn.call_count == 1
        # The project passed to burn_captions should have cleared captioned_clip_path
        project_arg = mock_burn.call_args[0][0]
        for clip in project_arg.clips:
            assert clip.captioned_clip_path is None


class TestReburnErrorHandling:
    """Error handling tests."""

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_skips_project_without_transcript(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama", llm_model="llama3.2", vocabulary={},
        )

        # Delete the transcript file
        (brand_dir / "transcripts" / "test-video.json").unlink()

        result = runner.invoke(app, ["re-burn-captions", "test-brand"])
        assert result.exit_code == 0
        assert "transcript not found" in result.output.lower() or "no transcript" in result.output.lower()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli_reburn.py -v`
Expected: FAIL — `re-burn-captions` command doesn't exist yet.

**Step 3: Implement the `re-burn-captions` command**

Add the following to `src/clip_video/cli.py` after the `refine` command (after the Refinement Complete panel, before the `_get_video_duration` helper or next command section).

The command needs to:

1. Find all `project_state.json` files under `brands/{brand}/highlights/`
2. Filter by `--project` substring match if provided
3. For each project:
   a. Load `HighlightsProject.load(state_file)`
   b. Determine the video stem from `project.video_path` and find the transcript
   c. Load `TranscriptionResult.load(transcript_path)` to get segments
   d. Back up existing final clips to `clips/backups/pre-reburn/` or `clips/backups/reburn-{timestamp}/`
   e. Clear `clip.captioned_clip_path = None` for each clip
   f. Create a `HighlightsProcessor` and call `burn_captions(project, transcript_segments)`
   g. Save project state
4. Print summary

Key implementation details:

```python
@app.command()
def re_burn_captions(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    project: Annotated[
        Optional[str],
        typer.Option("--project", "-p", help="Filter to projects matching this substring"),
    ] = None,
) -> None:
    """Re-burn captions on existing highlight clips using updated transcripts.

    After refining transcripts, use this to update captions on already-generated
    clips without re-extracting or re-analyzing.
    """
```

For the processor instantiation, create a minimal `HighlightsConfig` with the brand's LLM config (even though captions don't use the LLM, the processor constructor requires it):

```python
from clip_video.modes.highlights import HighlightsConfig, HighlightsProcessor, HighlightsProject
from clip_video.transcription.base import TranscriptionResult
from clip_video.llm.base import LLMConfig, LLMProviderType

llm_config = LLMConfig(provider=LLMProviderType(config.llm_provider), model=config.llm_model)
highlights_config = HighlightsConfig(llm_config=llm_config, brand_config=config)
processor = HighlightsProcessor(config=highlights_config)
```

For backup logic:

```python
backups_dir = project.clips_dir / "backups"
pre_reburn_dir = backups_dir / "pre-reburn"

if not pre_reburn_dir.exists():
    # First reburn — back up to pre-reburn/
    pre_reburn_dir.mkdir(parents=True)
    for clip in project.clips:
        if clip.captioned_clip_path and Path(clip.captioned_clip_path).exists():
            shutil.copy2(clip.captioned_clip_path, pre_reburn_dir / Path(clip.captioned_clip_path).name)
else:
    # Subsequent reburn — timestamped backup
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    ts_dir = backups_dir / f"reburn-{ts}"
    ts_dir.mkdir(parents=True)
    for clip in project.clips:
        if clip.captioned_clip_path and Path(clip.captioned_clip_path).exists():
            shutil.copy2(clip.captioned_clip_path, ts_dir / Path(clip.captioned_clip_path).name)
```

For clearing captioned paths and re-burning:

```python
# Clear captioned_clip_path so burn_captions doesn't skip
for clip in project.clips:
    clip.captioned_clip_path = None

# Re-burn captions with updated transcript
processor.burn_captions(project, transcript_result.segments)
```

For finding the transcript from the project's video_path:

```python
video_stem = Path(project.video_path).stem
transcript_path = transcripts_dir / f"{video_stem}.json"
if not transcript_path.exists():
    console.print(f"    [yellow]Transcript not found for {video_stem}, skipping[/yellow]")
    skipped += 1
    continue
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_reburn.py -v`
Expected: All tests PASS.

**Step 5: Run full test suite for regressions**

Run: `python -m pytest --tb=short`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add tests/test_cli_reburn.py src/clip_video/cli.py
git commit -m "feat: add re-burn-captions command for updating clips with refined transcripts"
```

---

### Task 2: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md**

Add a section for the `re-burn-captions` command near the existing `refine` and `highlights` documentation:

```markdown
### Re-burn captions on existing clips

After refining transcripts, update captions on already-generated highlight clips:

```bash
# Re-burn captions for all projects in a brand
clip-video re-burn-captions KCD-UK

# Re-burn a specific project
clip-video re-burn-captions KCD-UK --project "David Flanagan"
```

| Option | Description |
|---|---|
| `--project TEXT` | Filter to projects matching this substring |

Previous final clips are backed up to `clips/backups/pre-reburn/` on first run,
and `clips/backups/reburn-YYYY-MM-DDTHHMMSS/` on subsequent runs.
```

**Step 2: Update CLAUDE.md**

Add `clip-video re-burn-captions BRAND` to the Quick Reference section and mention the backup strategy in Important Implementation Details.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add re-burn-captions command to README and CLAUDE.md"
```
