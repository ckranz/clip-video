"""Command-line interface for clip-video.

Uses Typer for a modern, type-hinted CLI experience.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env files
# Priority: local .env > ~/.clip-video/.env
_user_env = Path.home() / ".clip-video" / ".env"
if _user_env.exists():
    load_dotenv(_user_env)
load_dotenv()  # Load local .env (overrides user-level)
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from clip_video import __version__
from clip_video.config import (
    BrandConfig,
    brand_exists,
    get_brand_path,
    list_brands,
    load_brand_config,
    save_brand_config,
)
from clip_video.ffmpeg_binary import (
    FFmpegConfig,
    get_bin_directory,
    get_dependency_report,
    get_ffmpeg_info,
    get_ffprobe_path,
    install_custom_ffmpeg,
    verify_ffmpeg,
)

# Create the main Typer app
app = typer.Typer(
    name="clip-video",
    help="Intelligent video clip extraction tool for lyric matching and highlights generation.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"clip-video version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Clip Video - Intelligent Video Clip Extraction Tool.

    Two primary modes:

    [bold]lyric-match[/bold]: Build searchable word/phrase dictionaries from video libraries
    for creating music video mashups where speakers "sing" songs.

    [bold]highlights[/bold]: Batch-process recordings to generate social media shorts
    with burned-in captions and brand-specific visual enhancements.
    """
    pass


# =============================================================================
# Brand Management Commands
# =============================================================================


@app.command()
def init_brand(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand to create")],
    description: Annotated[
        str, typer.Option("--description", "-d", help="Description of the brand")
    ] = "",
) -> None:
    """Initialize a new brand with folder structure.

    Creates the brand directory with subdirectories for videos, transcripts,
    projects, and outputs. Also creates a default configuration file.
    """
    if brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' already exists.")
        raise typer.Exit(1)

    brand_path = get_brand_path(brand_name)

    # Create directory structure
    directories = [
        brand_path / "videos",
        brand_path / "transcripts",
        brand_path / "projects",
        brand_path / "outputs",
        brand_path / "search_results",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Create default configuration
    config = BrandConfig(
        name=brand_name,
        description=description,
        # Include starter CNCF vocabulary
        vocabulary={
            "kubernetes": ["cooper netties", "kuber nettis", "kuber netties", "cooper nettis"],
            "argocd": ["argo cd", "argo seedy", "argo c d", "argo cede"],
            "istio": ["is theo", "east io", "isto", "is tio"],
            "etcd": ["et cetera d", "e t c d", "et cd", "etcetera d"],
            "prometheus": ["pro metheus"],
            "grafana": ["graffana", "graph ana"],
            "helm": ["hellm"],
            "containerd": ["container d", "container dee"],
            "envoy": ["en voy"],
            "jaeger": ["jager", "yaeger", "yager"],
            "fluentd": ["fluent d", "fluent dee"],
            "falco": ["fallco"],
            "linkerd": ["linker d", "linker dee"],
            "cncf": ["c n c f", "cnc f", "see ncf"],
        },
    )

    config_path = save_brand_config(brand_name, config)

    # Display success message
    console.print(
        Panel(
            f"[green]Brand '{brand_name}' created successfully![/green]\n\n"
            f"Location: {brand_path}\n\n"
            "Directory structure:\n"
            f"  {brand_path}/\n"
            "    videos/        - Place source videos here\n"
            "    transcripts/   - Generated transcripts\n"
            "    projects/      - Project configurations\n"
            "    outputs/       - Generated clips\n"
            "    search_results/ - Search result clips\n"
            f"    config.json    - Brand configuration\n\n"
            "Next steps:\n"
            "  1. Copy video files to the videos/ directory\n"
            "  2. Run: clip-video transcribe " + brand_name + "\n"
            "  3. Edit config.json to customize settings",
            title="Brand Initialized",
        )
    )


@app.command()
def list_brands_cmd() -> None:
    """List all available brands."""
    brands = list_brands()

    if not brands:
        console.print("[yellow]No brands found.[/yellow]")
        console.print("Create one with: clip-video init-brand <name>")
        return

    table = Table(title="Available Brands")
    table.add_column("Brand Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Videos", style="green")

    for brand_name in brands:
        try:
            config = load_brand_config(brand_name)
            brand_path = get_brand_path(brand_name)
            video_count = len(list((brand_path / "videos").glob("*")))
            table.add_row(brand_name, config.description or "-", str(video_count))
        except Exception:
            table.add_row(brand_name, "[red]Error loading config[/red]", "-")

    console.print(table)


# =============================================================================
# Transcription Commands
# =============================================================================


@app.command()
def transcribe(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand to transcribe")],
    video: Annotated[
        Optional[Path],
        typer.Option("--video", "-f", help="Specific video file to transcribe"),
    ] = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Transcription provider (whisper_api, whisper_local)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-transcribe even if transcript exists"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip cost confirmation prompt"),
    ] = False,
) -> None:
    """Transcribe videos for a brand.

    Processes all untranscribed videos in the brand's videos folder,
    or a specific video if --video is provided.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    # Import transcription modules
    from clip_video.transcription import (
        TranscriptionProgress,
        WhisperAPIProvider,
        WhisperLocalProvider,
    )
    from clip_video.vocabulary import VocabularyTerms, TranscriptCorrector

    # Load brand config
    config = load_brand_config(brand_name)
    brand_path = get_brand_path(brand_name)

    # Determine provider
    provider_name = provider or config.transcription_provider

    # Initialize provider
    if provider_name == "whisper_local":
        transcription_provider = WhisperLocalProvider()
        if not transcription_provider.is_available():
            console.print("[red]Error:[/red] Local Whisper not available.")
            console.print("Install faster-whisper: pip install faster-whisper")
            raise typer.Exit(1)
    else:
        # Default to API
        transcription_provider = WhisperAPIProvider()
        if not transcription_provider.is_available():
            console.print("[red]Error:[/red] OpenAI API not configured.")
            console.print("Set OPENAI_API_KEY environment variable.")
            raise typer.Exit(1)

    # Load vocabulary for corrections
    vocabulary = VocabularyTerms(config.vocabulary) if config.vocabulary else VocabularyTerms()
    corrector = TranscriptCorrector(vocabulary)
    whisper_prompt = vocabulary.generate_whisper_prompt()

    # Load progress tracker
    progress_path = brand_path / "transcripts" / ".transcription_progress.json"
    progress_tracker = TranscriptionProgress.load_or_create(progress_path, brand_name)
    progress_tracker.provider = provider_name

    # Get videos to process
    videos_dir = brand_path / "videos"
    transcripts_dir = brand_path / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # Video file extensions
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv"}

    if video:
        # Single video mode
        if not video.exists():
            console.print(f"[red]Error:[/red] Video file not found: {video}")
            raise typer.Exit(1)
        videos_to_process = [video]
    else:
        # Batch mode - find all videos
        videos_to_process = [
            f for f in videos_dir.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]

    if not videos_to_process:
        console.print("[yellow]No videos found to transcribe.[/yellow]")
        return

    # Filter based on existing transcripts and progress
    videos_needing_transcription = []
    for video_path in videos_to_process:
        transcript_path = transcripts_dir / f"{video_path.stem}.json"

        # Check if already completed (unless force)
        if not force:
            if transcript_path.exists():
                progress_tracker.skip_video(video_path, "Transcript exists")
                continue
            if progress_tracker.is_completed(str(video_path)):
                progress_tracker.skip_video(video_path, "Already transcribed")
                continue

        progress_tracker.add_video(video_path)
        videos_needing_transcription.append(video_path)

    if not videos_needing_transcription:
        console.print("[green]All videos already transcribed.[/green]")
        summary = progress_tracker.get_summary()
        console.print(f"Completed: {summary['completed']}, Skipped: {summary['skipped']}")
        return

    # Estimate total duration and cost
    total_duration = 0.0
    video_durations: dict[Path, float] = {}

    console.print(f"\n[cyan]Analyzing {len(videos_needing_transcription)} videos...[/cyan]")

    for video_path in videos_needing_transcription:
        duration = _get_video_duration(video_path)
        video_durations[video_path] = duration
        total_duration += duration

    # Show cost estimate for API provider
    estimated_cost = transcription_provider.estimate_cost(total_duration)

    console.print(Panel(
        f"[bold]Transcription Summary[/bold]\n\n"
        f"Videos to process: {len(videos_needing_transcription)}\n"
        f"Total duration: {total_duration / 60:.1f} minutes\n"
        f"Provider: {provider_name}\n"
        + (f"Estimated cost: ${estimated_cost:.2f} USD\n" if estimated_cost else "Cost: Free (local)\n")
        + f"Vocabulary terms: {len(vocabulary)}",
        title="Transcription Plan",
    ))

    # Confirm if using API and cost is significant
    if estimated_cost and estimated_cost > 0.01 and not yes:
        if not typer.confirm("Proceed with transcription?"):
            console.print("[yellow]Transcription cancelled.[/yellow]")
            raise typer.Exit(0)

    # Process videos
    console.print()
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task(
            "Transcribing...",
            total=len(videos_needing_transcription),
        )

        for video_path in videos_needing_transcription:
            progress_bar.update(task, description=f"Transcribing {video_path.name}...")

            try:
                # Mark as in progress
                progress_tracker.start_video(video_path)
                progress_tracker.save(progress_path)

                # Transcribe
                result = transcription_provider.transcribe(
                    video_path,
                    language="en",
                    prompt=whisper_prompt,
                )

                # Apply vocabulary corrections
                all_words = []
                for segment in result.segments:
                    for word in segment.words:
                        all_words.append(word.__dict__)

                corrected_words, correction_log = corrector.correct_words(
                    all_words,
                    source_file=str(video_path),
                )

                # Update words in segments
                word_idx = 0
                for segment in result.segments:
                    segment_word_count = len(segment.words)
                    for i in range(segment_word_count):
                        if word_idx < len(corrected_words):
                            segment.words[i].word = corrected_words[word_idx]["word"]
                            if "original_word" in corrected_words[word_idx]:
                                segment.words[i].original_word = corrected_words[word_idx]["original_word"]
                            word_idx += 1

                    # Update segment text from corrected words
                    segment.text = " ".join(w.word for w in segment.words)

                # Update full text
                result.text = " ".join(seg.text for seg in result.segments)
                result.vocabulary_corrections = len(correction_log)

                # Save transcript
                transcript_path = transcripts_dir / f"{video_path.stem}.json"
                result.save(transcript_path)

                # Update progress
                progress_tracker.complete_video(
                    video_path,
                    transcript_path,
                    duration_seconds=result.duration,
                    cost_usd=result.cost_usd,
                )
                progress_tracker.save(progress_path)

                successful += 1

            except Exception as e:
                progress_tracker.fail_video(video_path, str(e))
                progress_tracker.save(progress_path)
                failed += 1
                console.print(f"\n[red]Error transcribing {video_path.name}:[/red] {e}")

            progress_bar.advance(task)

    # Final summary
    console.print()
    summary = progress_tracker.get_summary()

    console.print(Panel(
        f"[bold]Transcription Complete[/bold]\n\n"
        f"Successful: [green]{successful}[/green]\n"
        f"Failed: [red]{failed}[/red]\n"
        f"Total processed: {summary['completed']}\n"
        + (f"Total cost: ${summary['total_cost_usd']:.2f} USD" if summary['total_cost_usd'] else ""),
        title="Results",
    ))


def _get_video_duration(video_path: Path) -> float:
    """Get duration of a video file in seconds."""
    import subprocess

    ffprobe_path = get_ffprobe_path()
    if not ffprobe_path:
        return 0.0

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


# =============================================================================
# Search Commands
# =============================================================================


@app.command()
def search(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand to search")],
    phrase: Annotated[str, typer.Argument(help="Word or phrase to search for")],
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of results"),
    ] = 10,
    export: Annotated[
        bool,
        typer.Option("--export", "-e", help="Export matching clips to search_results folder"),
    ] = False,
) -> None:
    """Search for a word or phrase across all transcripts.

    Finds all occurrences of the search term in the brand's video library
    and optionally exports preview clips.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    # TODO: Implement search logic in task 10
    console.print(f"[yellow]Search not yet implemented.[/yellow]")
    console.print(f"Brand: {brand_name}")
    console.print(f"Phrase: '{phrase}'")
    console.print(f"Limit: {limit}")


# =============================================================================
# Lyric Match Mode
# =============================================================================


@app.command()
def lyric_match(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    project_name: Annotated[str, typer.Argument(help="Name of the project")],
    lyrics_file: Annotated[Path, typer.Argument(help="Path to lyrics text file")],
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume existing project"),
    ] = False,
) -> None:
    """Start a lyric match project.

    Parses the lyrics file and extracts candidate clips for each word/phrase
    from the brand's video library.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    if not lyrics_file.exists():
        console.print(f"[red]Error:[/red] Lyrics file not found: {lyrics_file}")
        raise typer.Exit(1)

    # TODO: Implement lyric match logic in task 11
    console.print(f"[yellow]Lyric match mode not yet implemented.[/yellow]")
    console.print(f"Brand: {brand_name}")
    console.print(f"Project: {project_name}")
    console.print(f"Lyrics: {lyrics_file}")


# =============================================================================
# Highlights Mode
# =============================================================================


@app.command()
def highlights(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    video: Annotated[Path, typer.Argument(help="Path to video file")],
    description: Annotated[
        Optional[Path],
        typer.Option("--description", "-d", help="Path to session description file"),
    ] = None,
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of highlights to generate"),
    ] = 5,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
) -> None:
    """Generate highlight clips from a video.

    Analyzes the video transcript, identifies highlight-worthy segments,
    and generates portrait-format clips with burned-in captions.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    if not video.exists():
        console.print(f"[red]Error:[/red] Video file not found: {video}")
        raise typer.Exit(1)

    from clip_video.modes.highlights import (
        HighlightsConfig,
        HighlightsProcessor,
        HighlightsProject,
    )
    from clip_video.transcription import TranscriptionResult

    brand_path = get_brand_path(brand_name)
    config = load_brand_config(brand_name)

    # Check for existing transcript
    transcript_path = brand_path / "transcripts" / f"{video.stem}.json"
    if not transcript_path.exists():
        console.print(f"[red]Error:[/red] No transcript found for {video.name}")
        console.print(f"Run: clip-video transcribe {brand_name} --video {video}")
        raise typer.Exit(1)

    # Load transcript
    transcript_result = TranscriptionResult.load(transcript_path)

    # Create highlights config
    highlights_config = HighlightsConfig(target_clips=count)

    # Create processor with progress callback
    def progress_callback(stage: str, progress: float) -> None:
        pass  # Progress shown via rich progress bar

    processor = HighlightsProcessor(config=highlights_config, progress_callback=progress_callback)

    # Estimate cost
    transcript_text = "\n".join(
        f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}"
        for seg in transcript_result.segments
    )
    estimated_cost = processor.get_cost_estimate(transcript_text)

    console.print(Panel(
        f"[bold]Highlights Generation[/bold]\n\n"
        f"Video: {video.name}\n"
        f"Target clips: {count}\n"
        f"LLM Provider: {config.llm_provider}\n"
        f"Estimated LLM cost: ${estimated_cost:.3f} USD",
        title="Highlights Plan",
    ))

    if not yes and estimated_cost > 0.01:
        if not typer.confirm("Proceed with highlights generation?"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    # Create project
    project_name = f"{video.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_root = brand_path / "highlights" / project_name

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Creating project...", total=6)

        # Create project
        project = processor.create_project(
            name=project_name,
            brand_name=brand_name,
            video_path=video,
            description_path=description,
            project_root=project_root,
        )
        progress_bar.update(task, advance=1, description="Analyzing transcript...")

        # Set transcript
        project.transcript_text = transcript_text
        project.save()

        # Analyze
        try:
            analysis = processor.analyze(project)
            progress_bar.update(task, advance=1, description=f"Found {len(analysis.segments)} highlights...")
        except Exception as e:
            console.print(f"\n[red]Error during analysis:[/red] {e}")
            raise typer.Exit(1)

        if not analysis.segments:
            console.print("\n[yellow]No highlight segments identified.[/yellow]")
            raise typer.Exit(0)

        # Extract clips
        progress_bar.update(task, description="Extracting clips...")
        try:
            processor.extract_clips(project)
            progress_bar.update(task, advance=1, description="Converting to portrait...")
        except Exception as e:
            console.print(f"\n[red]Error extracting clips:[/red] {e}")
            raise typer.Exit(1)

        # Convert to portrait
        try:
            processor.convert_to_portrait(project)
            progress_bar.update(task, advance=1, description="Burning captions...")
        except Exception as e:
            console.print(f"\n[red]Error converting to portrait:[/red] {e}")
            raise typer.Exit(1)

        # Burn captions
        try:
            processor.burn_captions(project, transcript_result.segments)
            progress_bar.update(task, advance=1, description="Generating metadata...")
        except Exception as e:
            console.print(f"\n[red]Error burning captions:[/red] {e}")
            raise typer.Exit(1)

        # Generate metadata
        try:
            processor.generate_metadata(project)
            progress_bar.update(task, advance=1, description="Complete!")
        except Exception as e:
            console.print(f"\n[red]Error generating metadata:[/red] {e}")
            raise typer.Exit(1)

    # Show results
    console.print()
    console.print(Panel(
        f"[bold green]Highlights Generated![/bold green]\n\n"
        f"Clips created: {len(project.clips)}\n"
        f"Output folder: {project.final_clips_dir}\n\n"
        + "\n".join(
            f"  {clip.clip_id}: {clip.segment.summary[:50]}..."
            for clip in project.clips[:5]
        )
        + ("\n  ..." if len(project.clips) > 5 else ""),
        title="Results",
    ))


@app.command()
def highlights_batch(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    video_list: Annotated[Path, typer.Argument(help="Path to file listing videos to process (or directory)")],
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of highlights per video"),
    ] = 5,
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume an existing batch job"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
    parallel: Annotated[
        int,
        typer.Option("--parallel", "-p", help="Number of parallel workers (1 = sequential)"),
    ] = 1,
) -> None:
    """Batch process multiple videos for highlights.

    Reads a list of video files (one per line) or a directory and processes
    each one to generate highlight clips. Progress is saved for resume capability.

    The video list file should contain video filenames (one per line).
    Videos are expected to be in the brand's videos/ folder.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    if not video_list.exists():
        console.print(f"[red]Error:[/red] Video list file not found: {video_list}")
        raise typer.Exit(1)

    from clip_video.batch import BatchConfig, BatchProcessor, BatchJob
    from clip_video.modes.highlights import HighlightsConfig
    from clip_video.transcription import TranscriptionResult

    brand_path = get_brand_path(brand_name)
    videos_dir = brand_path / "videos"
    transcripts_dir = brand_path / "transcripts"

    # Load video list
    if video_list.is_dir():
        video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
        video_paths = [f for f in video_list.iterdir() if f.suffix.lower() in video_extensions]
    else:
        # Read video list file - each line is a filename
        lines = video_list.read_text(encoding="utf-8").strip().splitlines()
        video_paths = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Check if it's a full path or just a filename
            video_path = Path(line)
            if not video_path.is_absolute():
                video_path = videos_dir / line
            if video_path.exists():
                video_paths.append(video_path)
            else:
                console.print(f"[yellow]Warning:[/yellow] Video not found: {line}")

    if not video_paths:
        console.print("[red]Error:[/red] No valid videos found in the list.")
        raise typer.Exit(1)

    # Check which videos have transcripts
    videos_with_transcripts = []
    videos_without_transcripts = []
    for vp in video_paths:
        transcript_path = transcripts_dir / f"{vp.stem}.json"
        if transcript_path.exists():
            videos_with_transcripts.append(vp)
        else:
            videos_without_transcripts.append(vp)

    if videos_without_transcripts:
        console.print(f"\n[yellow]Warning:[/yellow] {len(videos_without_transcripts)} videos missing transcripts:")
        for vp in videos_without_transcripts[:5]:
            console.print(f"  - {vp.name}")
        if len(videos_without_transcripts) > 5:
            console.print(f"  ... and {len(videos_without_transcripts) - 5} more")
        console.print(f"\nRun: clip-video transcribe {brand_name}")

    if not videos_with_transcripts:
        console.print("\n[red]Error:[/red] No videos have transcripts. Transcribe first.")
        raise typer.Exit(1)

    # Create batch job name
    job_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_root = brand_path / "batch_jobs" / job_name

    # Check for existing job to resume
    if resume:
        existing_jobs = list((brand_path / "batch_jobs").glob("*/batch_state.json"))
        if existing_jobs:
            # Load most recent job
            latest_job_file = max(existing_jobs, key=lambda p: p.stat().st_mtime)
            console.print(f"[cyan]Resuming batch job:[/cyan] {latest_job_file.parent.name}")
            batch_config = BatchConfig(
                max_parallel=parallel,
                continue_on_error=True,
                skip_completed=True,
                highlights_config=HighlightsConfig(target_clips=count),
            )
            job = BatchJob.load(latest_job_file, batch_config)
        else:
            console.print("[yellow]No existing batch job found. Starting new batch.[/yellow]")
            resume = False

    if not resume:
        # Show plan
        console.print(Panel(
            f"[bold]Batch Highlights Generation[/bold]\n\n"
            f"Videos to process: {len(videos_with_transcripts)}\n"
            f"Highlights per video: {count}\n"
            f"Parallel workers: {parallel}\n"
            f"Output folder: {brand_path / 'highlights'}",
            title="Batch Plan",
        ))

        if not yes:
            if not typer.confirm("Proceed with batch processing?"):
                console.print("[yellow]Cancelled.[/yellow]")
                raise typer.Exit(0)

        # Create batch config and job
        batch_config = BatchConfig(
            max_parallel=parallel,
            continue_on_error=True,
            skip_completed=True,
            highlights_config=HighlightsConfig(target_clips=count),
        )

        processor = BatchProcessor(config=batch_config)
        job = processor.create_job(
            name=job_name,
            brand_name=brand_name,
            video_paths=videos_with_transcripts,
            job_root=job_root,
        )

    # Process with progress display
    processor = BatchProcessor(config=batch_config)

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task(
            "Processing videos...",
            total=len(videos_with_transcripts),
        )

        # Process each video
        pending = job.get_pending_videos()
        for i, video_path in enumerate(pending):
            progress_bar.update(
                task,
                description=f"Processing {video_path.name}...",
                completed=job.completed_count,
            )

            # Load transcript
            transcript_path = transcripts_dir / f"{video_path.stem}.json"
            try:
                transcript_result = TranscriptionResult.load(transcript_path)
            except Exception as e:
                # Mark as failed and continue
                result = job.results.get(str(video_path))
                if result:
                    result.status = BatchProcessor.__module__  # Use VideoStatus
                    from clip_video.batch import VideoStatus
                    result.status = VideoStatus.FAILED
                    result.error_message = f"Failed to load transcript: {e}"
                job.save()
                continue

            # Process with the single video highlights logic
            try:
                from clip_video.modes.highlights import HighlightsProcessor, HighlightsConfig as HC
                from clip_video.batch import VideoStatus

                highlights_config = HC(target_clips=count)
                hl_processor = HighlightsProcessor(config=highlights_config)

                project_name = f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_root = brand_path / "highlights" / project_name

                project = hl_processor.create_project(
                    name=project_name,
                    brand_name=brand_name,
                    video_path=video_path,
                    project_root=project_root,
                )

                # Set transcript
                transcript_text = "\n".join(
                    f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}"
                    for seg in transcript_result.segments
                )
                project.transcript_text = transcript_text
                project.save()

                # Run pipeline
                hl_processor.analyze(project)
                hl_processor.extract_clips(project)
                hl_processor.convert_to_portrait(project)
                hl_processor.burn_captions(project, transcript_result.segments)
                hl_processor.generate_metadata(project)

                # Update job result
                result = job.results.get(str(video_path))
                if result:
                    result.status = VideoStatus.COMPLETED
                    result.project_name = project_name
                    result.clips_generated = len(project.clips)
                    result.clip_paths = [
                        str(c.final_clip_path) for c in project.clips if c.final_clip_path
                    ]
                    result.completed_at = datetime.now().isoformat()

            except Exception as e:
                from clip_video.batch import VideoStatus
                result = job.results.get(str(video_path))
                if result:
                    result.status = VideoStatus.FAILED
                    result.error_message = str(e)
                    result.completed_at = datetime.now().isoformat()
                console.print(f"\n[red]Error processing {video_path.name}:[/red] {e}")

            job.save()
            progress_bar.update(task, completed=job.completed_count + job.failed_count)

    # Generate report
    report = job.generate_report()

    # Show results
    console.print()
    console.print(Panel(
        f"[bold]Batch Complete[/bold]\n\n"
        f"Total videos: {job.total_videos}\n"
        f"Completed: [green]{job.completed_count}[/green]\n"
        f"Failed: [red]{job.failed_count}[/red]\n"
        f"Total clips generated: {job.total_clips_generated}\n\n"
        f"Report saved: {job.report_file}",
        title="Results",
    ))

    if job.failed_count > 0:
        console.print("\n[yellow]Failed videos:[/yellow]")
        for path, error in job.get_failed_videos()[:5]:
            console.print(f"  - {path.name}: {error[:50]}...")


# =============================================================================
# Utility Commands
# =============================================================================


@app.command()
def check_deps(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed dependency information"),
    ] = False,
    custom_ffmpeg: Annotated[
        Optional[Path],
        typer.Option("--set-ffmpeg", help="Install custom FFmpeg binary from path"),
    ] = None,
    custom_ffprobe: Annotated[
        Optional[Path],
        typer.Option("--set-ffprobe", help="Install custom FFprobe binary from path"),
    ] = None,
) -> None:
    """Check and report on required dependencies.

    Verifies FFmpeg is available and reports version information.
    Use --set-ffmpeg to install a custom FFmpeg binary.
    """
    # Handle custom binary installation
    if custom_ffmpeg:
        console.print(f"[cyan]Installing custom FFmpeg from:[/cyan] {custom_ffmpeg}")
        success, message = install_custom_ffmpeg(
            str(custom_ffmpeg),
            str(custom_ffprobe) if custom_ffprobe else None
        )
        if success:
            console.print(f"[green]{message}[/green]")
        else:
            console.print(f"[red]Error:[/red] {message}")
            raise typer.Exit(1)
        console.print()

    # Get FFmpeg information
    ffmpeg_info = get_ffmpeg_info()
    success, message = verify_ffmpeg()

    # Build status display
    if verbose:
        # Detailed report
        report = get_dependency_report()

        table = Table(title="Dependency Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        # FFmpeg
        if ffmpeg_info.available:
            table.add_row(
                "FFmpeg",
                f"[green]Available[/green] (v{ffmpeg_info.version})",
                f"Source: {ffmpeg_info.source}\n{ffmpeg_info.path}"
            )
        else:
            table.add_row(
                "FFmpeg",
                "[red]Not Found[/red]",
                "Install with: pip install imageio-ffmpeg"
            )

        # FFprobe
        ffprobe_info = report.get("ffprobe", {})
        if ffprobe_info.get("available"):
            table.add_row(
                "FFprobe",
                "[green]Available[/green]",
                str(ffprobe_info.get("path", ""))
            )
        else:
            table.add_row(
                "FFprobe",
                "[yellow]Not Found[/yellow]",
                "Optional - used for media inspection"
            )

        # imageio-ffmpeg
        imageio_info = report.get("imageio_ffmpeg", {})
        if imageio_info.get("available"):
            table.add_row(
                "imageio-ffmpeg",
                f"[green]Installed[/green]",
                f"Version: {imageio_info.get('version', 'unknown')}"
            )
        else:
            table.add_row(
                "imageio-ffmpeg",
                "[yellow]Not Installed[/yellow]",
                "Recommended: pip install imageio-ffmpeg"
            )

        # Platform info
        platform_info = report.get("platform", {})
        table.add_row(
            "Platform",
            str(platform_info.get("system", "")),
            str(platform_info.get("machine", ""))
        )

        console.print(table)
        console.print()
        console.print(f"[dim]Custom binary location: {get_bin_directory()}[/dim]")

    else:
        # Simple status
        console.print(Panel(
            f"[bold]Dependency Check[/bold]\n\n" +
            (f"[green]FFmpeg:[/green] v{ffmpeg_info.version} ({ffmpeg_info.source})\n"
             f"  [dim]{ffmpeg_info.path}[/dim]"
             if ffmpeg_info.available
             else "[red]FFmpeg:[/red] Not found\n"
                  "  Install with: pip install imageio-ffmpeg"),
            title="clip-video dependencies",
        ))

    # Exit with error if FFmpeg not available
    if not ffmpeg_info.available:
        console.print("\n[yellow]Tip:[/yellow] Run 'pip install imageio-ffmpeg' to auto-download FFmpeg")
        raise typer.Exit(1)


@app.command()
def info(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
) -> None:
    """Show detailed information about a brand."""
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    try:
        config = load_brand_config(brand_name)
        brand_path = get_brand_path(brand_name)

        # Count items
        video_count = len(list((brand_path / "videos").glob("*")))
        transcript_count = len(list((brand_path / "transcripts").glob("*.json")))
        project_count = len(list((brand_path / "projects").iterdir())) if (brand_path / "projects").exists() else 0

        console.print(Panel(
            f"[bold]{config.name}[/bold]\n"
            f"{config.description or 'No description'}\n\n"
            f"[cyan]Location:[/cyan] {brand_path}\n"
            f"[cyan]Videos:[/cyan] {video_count}\n"
            f"[cyan]Transcripts:[/cyan] {transcript_count}\n"
            f"[cyan]Projects:[/cyan] {project_count}\n\n"
            f"[cyan]Transcription Provider:[/cyan] {config.transcription_provider}\n"
            f"[cyan]LLM Provider:[/cyan] {config.llm_provider}\n"
            f"[cyan]Vocabulary Terms:[/cyan] {len(config.vocabulary)}",
            title="Brand Information",
        ))
    except Exception as e:
        console.print(f"[red]Error loading brand:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
