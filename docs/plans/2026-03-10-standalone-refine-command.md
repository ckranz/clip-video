# Standalone `refine` Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `clip-video refine BRAND` command that runs LLM refinement on existing transcripts without re-running Whisper.

**Architecture:** New CLI command loads transcript JSON files via `TranscriptionResult.load()`, runs `TranscriptRefiner.refine()` on each, backs up originals, and saves corrected versions in-place. Reuses all existing infrastructure — no new modules.

**Tech Stack:** Typer CLI, TranscriptRefiner, TranscriptionResult, Rich console output.

---

### Task 1: Core `refine` command — backup and refinement logic

**Files:**
- Modify: `src/clip_video/cli.py` (add new command after line 801)
- Test: `tests/test_cli_refine.py` (create)

**Step 1: Write the failing test for basic refine command**

Create `tests/test_cli_refine.py`:

```python
"""Tests for the standalone refine CLI command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from clip_video.cli import app
from clip_video.transcription.base import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)

runner = CliRunner()


def _make_transcript(video_name: str, text: str = "hello world") -> TranscriptionResult:
    """Create a minimal TranscriptionResult for testing."""
    return TranscriptionResult(
        video_path=f"/videos/{video_name}.mp4",
        text=text,
        segments=[
            TranscriptionSegment(
                text=text,
                start=0.0,
                end=2.0,
                words=[
                    TranscriptionWord(word=w, start=i * 0.5, end=(i + 1) * 0.5)
                    for i, w in enumerate(text.split())
                ],
            )
        ],
        provider="whisper_local",
        model="medium",
        duration=2.0,
    )


@pytest.fixture
def brand_dir(tmp_path):
    """Set up a fake brand with transcripts."""
    brand_path = tmp_path / "brands" / "test-brand"
    transcripts_dir = brand_path / "transcripts"
    transcripts_dir.mkdir(parents=True)

    # Save two transcripts
    t1 = _make_transcript("talk-one", "cooper netties are great")
    t1.save(transcripts_dir / "talk-one.json")

    t2 = _make_transcript("talk-two", "this is normal text")
    t2.save(transcripts_dir / "talk-two.json")

    # Create minimal config
    config_path = brand_path / "config.json"
    config_path.write_text(json.dumps({
        "llm_provider": "ollama",
        "llm_model": "llama3.2",
    }))

    return brand_path


class TestRefineCommand:
    """Tests for clip-video refine command."""

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_processes_all_transcripts(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        corrections_json = json.dumps([
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "domain term"}
        ])

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=True,
        ), patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner._call_llm",
            return_value=corrections_json,
        ):
            result = runner.invoke(app, ["refine", "test-brand"])

        assert result.exit_code == 0
        assert "talk-one" in result.output
        assert "talk-two" in result.output

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_creates_pre_refine_backup(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        corrections_json = json.dumps([
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "domain term"}
        ])

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=True,
        ), patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner._call_llm",
            return_value=corrections_json,
        ):
            result = runner.invoke(app, ["refine", "test-brand"])

        assert result.exit_code == 0
        # Pre-refine backup should exist for the transcript that had corrections
        backup = brand_dir / "transcripts" / "talk-one.pre-refine.json"
        assert backup.exists()

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_creates_timestamped_backup_on_rerun(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        # Create a pre-existing .pre-refine.json to simulate previous run
        transcripts_dir = brand_dir / "transcripts"
        shutil.copy(
            transcripts_dir / "talk-one.json",
            transcripts_dir / "talk-one.pre-refine.json",
        )

        corrections_json = json.dumps([
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "domain term"}
        ])

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=True,
        ), patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner._call_llm",
            return_value=corrections_json,
        ):
            result = runner.invoke(app, ["refine", "test-brand"])

        assert result.exit_code == 0
        # Should have a timestamped backup now
        backups = list(transcripts_dir.glob("talk-one.refine-*.json"))
        assert len(backups) == 1

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_video_filter(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        corrections_json = json.dumps([
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "domain term"}
        ])

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=True,
        ), patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner._call_llm",
            return_value=corrections_json,
        ):
            result = runner.invoke(app, ["refine", "test-brand", "--video", "talk-one"])

        assert result.exit_code == 0
        assert "talk-one" in result.output
        # talk-two should not be mentioned as processed
        assert "talk-two" not in result.output or "Skipping" not in result.output

    @patch("clip_video.cli.brand_exists", return_value=False)
    def test_refine_nonexistent_brand(self, mock_exists):
        result = runner.invoke(app, ["refine", "no-such-brand"])
        assert result.exit_code == 1

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_no_transcripts(
        self, mock_config, mock_exists, mock_path, tmp_path
    ):
        brand_path = tmp_path / "brands" / "empty-brand"
        transcripts_dir = brand_path / "transcripts"
        transcripts_dir.mkdir(parents=True)

        mock_path.return_value = brand_path
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        result = runner.invoke(app, ["refine", "empty-brand"])
        assert result.exit_code == 0
        assert "No transcripts found" in result.output

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_provider_not_available(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="claude",
            llm_model=None,
            vocabulary={},
            refine_transcripts=False,
        )

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=False,
        ):
            result = runner.invoke(app, ["refine", "test-brand"])

        assert result.exit_code == 1
        assert "not available" in result.output

    @patch("clip_video.cli.get_brand_path")
    @patch("clip_video.cli.brand_exists", return_value=True)
    @patch("clip_video.cli.load_brand_config")
    def test_refine_continues_on_individual_error(
        self, mock_config, mock_exists, mock_path, brand_dir
    ):
        mock_path.return_value = brand_dir
        mock_config.return_value = MagicMock(
            llm_provider="ollama",
            llm_model="llama3.2",
            vocabulary={},
            refine_transcripts=False,
        )

        call_count = 0

        def mock_call_llm(self_inner, system, user):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM temporarily unavailable")
            return json.dumps([])

        with patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner.is_available",
            return_value=True,
        ), patch(
            "clip_video.transcription.llm_refine.TranscriptRefiner._call_llm",
            mock_call_llm,
        ):
            result = runner.invoke(app, ["refine", "test-brand"])

        # Should complete despite one failure
        assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_refine.py -v`
Expected: FAIL — `refine` command doesn't exist yet.

**Step 3: Implement the `refine` command**

Add the following to `src/clip_video/cli.py` after line 801 (after the `transcribe` command's final summary panel):

```python
@app.command()
def refine(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand to refine transcripts for")],
    video: Annotated[
        Optional[str],
        typer.Option("--video", "-v", help="Filter to transcripts matching this substring"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="LLM provider for refinement (claude, openai, ollama)"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="LLM model for refinement"),
    ] = None,
    talk_title: Annotated[
        Optional[str],
        typer.Option("--talk-title", help="Talk title for LLM refinement context"),
    ] = None,
    talk_description: Annotated[
        Optional[str],
        typer.Option("--talk-description", help="Talk description for LLM refinement context"),
    ] = None,
) -> None:
    """Refine existing transcripts using LLM post-correction.

    Runs LLM refinement on already-transcribed videos to fix domain-specific
    terms, acronyms, proper nouns, and grammar errors without re-running Whisper.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    from clip_video.transcription.base import TranscriptionResult
    from clip_video.transcription.llm_refine import TranscriptRefiner, RefinementContext
    from clip_video.llm.base import LLMConfig, LLMProviderType
    from clip_video.vocabulary import VocabularyTerms

    # Load brand config
    config = load_brand_config(brand_name)
    brand_path = get_brand_path(brand_name)
    transcripts_dir = brand_path / "transcripts"

    # Find transcript files
    transcript_files = sorted(transcripts_dir.glob("*.json"))
    # Exclude non-transcript files (progress tracker, backups)
    transcript_files = [
        f for f in transcript_files
        if not f.name.startswith(".")
        and ".pre-refine." not in f.name
        and ".refine-" not in f.name
    ]

    # Apply video filter
    if video:
        transcript_files = [
            f for f in transcript_files
            if video.lower() in f.stem.lower()
        ]

    if not transcript_files:
        console.print("[yellow]No transcripts found to refine.[/yellow]")
        raise typer.Exit(0)

    # Set up LLM provider
    rp = provider or config.llm_provider
    rm = model or config.llm_model
    llm_config = LLMConfig(provider=LLMProviderType(rp), model=rm)
    refiner = TranscriptRefiner(llm_config)

    if not refiner.is_available():
        console.print(f"[red]Error:[/red] LLM provider '{rp}' is not available.")
        raise typer.Exit(1)

    # Load vocabulary for context
    vocabulary = VocabularyTerms(config.vocabulary) if config.vocabulary else VocabularyTerms()

    console.print(f"[bold]Refining {len(transcript_files)} transcript(s) for '{brand_name}'[/bold]")
    console.print(f"[dim]Provider: {rp} | Model: {rm or 'default'}[/dim]\n")

    total_corrections = 0
    refined_count = 0
    failed_count = 0

    for transcript_path in transcript_files:
        name = transcript_path.stem
        console.print(f"  Processing: {name}...")

        try:
            result = TranscriptionResult.load(transcript_path)

            # Build context
            ctx = RefinementContext(
                talk_title=talk_title or name,
                talk_description=talk_description,
                vocabulary_terms=vocabulary.get_all_terms() if vocabulary else None,
            )

            # Run refinement
            refined_segments, refine_log = refiner.refine(result.segments, context=ctx)
            correction_count = len(refine_log)

            if correction_count > 0:
                # Back up before overwriting
                pre_refine_path = transcript_path.with_suffix(".pre-refine.json")
                if not pre_refine_path.exists():
                    # First refinement — save original
                    shutil.copy2(transcript_path, pre_refine_path)
                else:
                    # Subsequent refinement — timestamped backup
                    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
                    backup_name = f"{transcript_path.stem}.refine-{ts}.json"
                    backup_path = transcript_path.parent / backup_name
                    shutil.copy2(transcript_path, backup_path)

                # Apply refinements
                result.segments = refined_segments
                result.text = " ".join(seg.text for seg in result.segments)
                result.vocabulary_corrections += correction_count
                result.save(transcript_path)

                total_corrections += correction_count
                refined_count += 1
                console.print(f"    [green]{correction_count} correction(s)[/green]")
            else:
                console.print(f"    [dim]No corrections needed[/dim]")

        except Exception as e:
            failed_count += 1
            console.print(f"    [red]Error: {e}[/red]")

    # Summary
    console.print()
    console.print(Panel(
        f"[bold]Refinement Complete[/bold]\n\n"
        f"Transcripts refined: [green]{refined_count}[/green]\n"
        f"No changes needed: {len(transcript_files) - refined_count - failed_count}\n"
        f"Failed: [red]{failed_count}[/red]\n"
        f"Total corrections: [bold]{total_corrections}[/bold]",
        title="Results",
    ))
```

Note: Also add `import shutil` to the top of `cli.py` (near line 8, with the other stdlib imports).

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_refine.py -v`
Expected: All 8 tests PASS.

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest --tb=short`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add tests/test_cli_refine.py src/clip_video/cli.py
git commit -m "feat: add standalone refine command for existing transcripts"
```

---

### Task 2: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md**

Add a new row to the CLI commands table for the `refine` command. Add a section describing usage, e.g.:

```markdown
### Refine existing transcripts

Run LLM refinement on already-transcribed videos without re-running Whisper:

```bash
# Refine all transcripts for a brand
clip-video refine KCD-UK --provider claude

# Refine a specific talk
clip-video refine KCD-UK --video "David Flanagan" --talk-title "Extreme Microservices"

# Use local Ollama (free)
clip-video refine KCD-UK --provider ollama --model llama3.2
```

| Option | Description |
|---|---|
| `--video TEXT` | Filter to transcripts matching this substring |
| `--provider TEXT` | LLM provider (claude, openai, ollama) |
| `--model TEXT` | LLM model override |
| `--talk-title TEXT` | Talk title for context |
| `--talk-description TEXT` | Talk description for context |
```

**Step 2: Update CLAUDE.md**

Add `refine` to the Quick Reference section and mention the backup strategy in the Important Implementation Details section.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add refine command to README and CLAUDE.md"
```
