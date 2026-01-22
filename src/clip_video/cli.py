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
from clip_video.review.queue import ReviewQueue, RejectedClip

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


def get_llm_provider_with_fallback(
    brand_name: str,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str | None]:
    """Get the LLM provider to use, with fallback logic.

    Checks provider availability and falls back to alternatives if needed.

    Args:
        brand_name: Name of the brand to check config for
        provider_override: CLI override for provider
        model_override: CLI override for model

    Returns:
        Tuple of (provider_name, model_name)

    Raises:
        typer.Exit: If no LLM provider is available
    """
    from clip_video.llm import ClaudeLLM, OpenAILLM, OllamaLLM, LLMConfig, LLMProviderType

    config = load_brand_config(brand_name)
    provider_name = provider_override or config.llm_provider
    model_name = model_override or config.llm_model

    # Check availability of each provider
    claude_available = os.environ.get("ANTHROPIC_API_KEY") is not None
    openai_available = os.environ.get("OPENAI_API_KEY") is not None
    ollama_available = OllamaLLM().is_available()

    if provider_name == "claude":
        if claude_available:
            return "claude", model_name
        elif ollama_available:
            console.print("[yellow]Warning:[/yellow] ANTHROPIC_API_KEY not set, falling back to Ollama (local).")
            return "ollama", model_name if model_name else "llama3.2"
        elif openai_available:
            console.print("[yellow]Warning:[/yellow] ANTHROPIC_API_KEY not set, falling back to OpenAI.")
            return "openai", model_name
        else:
            console.print("[red]Error:[/red] No LLM provider available.")
            console.print("Options:")
            console.print("  - Set ANTHROPIC_API_KEY for Claude")
            console.print("  - Set OPENAI_API_KEY for OpenAI")
            console.print("  - Install and run Ollama for free local inference: https://ollama.ai")
            raise typer.Exit(1)

    elif provider_name == "openai":
        if openai_available:
            return "openai", model_name
        elif ollama_available:
            console.print("[yellow]Warning:[/yellow] OPENAI_API_KEY not set, falling back to Ollama (local).")
            return "ollama", model_name if model_name else "llama3.2"
        elif claude_available:
            console.print("[yellow]Warning:[/yellow] OPENAI_API_KEY not set, falling back to Claude.")
            return "claude", model_name
        else:
            console.print("[red]Error:[/red] No LLM provider available.")
            console.print("Options:")
            console.print("  - Set OPENAI_API_KEY for OpenAI")
            console.print("  - Set ANTHROPIC_API_KEY for Claude")
            console.print("  - Install and run Ollama for free local inference: https://ollama.ai")
            raise typer.Exit(1)

    elif provider_name == "ollama":
        if ollama_available:
            return "ollama", model_name
        elif claude_available:
            console.print("[yellow]Warning:[/yellow] Ollama not running, falling back to Claude API.")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
            return "claude", model_name
        elif openai_available:
            console.print("[yellow]Warning:[/yellow] Ollama not running, falling back to OpenAI API.")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
            return "openai", model_name
        else:
            console.print("[red]Error:[/red] No LLM provider available.")
            console.print("Options:")
            console.print("  - Start Ollama: ollama serve")
            console.print("  - Set ANTHROPIC_API_KEY for Claude")
            console.print("  - Set OPENAI_API_KEY for OpenAI")
            raise typer.Exit(1)

    else:
        console.print(f"[red]Error:[/red] Unknown LLM provider '{provider_name}'.")
        console.print("Use 'claude', 'openai', or 'ollama'.")
        raise typer.Exit(1)


def check_llm_api_key(brand_name: str) -> None:
    """Check that the required LLM API key is configured.

    DEPRECATED: Use get_llm_provider_with_fallback instead.
    Kept for backwards compatibility.
    """
    get_llm_provider_with_fallback(brand_name)


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
    non_interactive: Annotated[
        bool, typer.Option("--yes", "-y", help="Use defaults without prompting")
    ] = False,
) -> None:
    """Initialize a new brand with folder structure.

    Creates the brand directory with subdirectories for videos, transcripts,
    projects, and outputs. Also creates a default configuration file.

    Prompts for key settings (transcription and LLM providers) unless --yes is used.
    """
    if brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' already exists.")
        raise typer.Exit(1)

    brand_path = get_brand_path(brand_name)

    # Check what providers are available
    from clip_video.llm.ollama import OllamaLLM
    from clip_video.transcription import WhisperLocalProvider

    ollama_available = OllamaLLM().is_available()
    anthropic_key_set = os.environ.get("ANTHROPIC_API_KEY") is not None
    openai_key_set = os.environ.get("OPENAI_API_KEY") is not None

    # Check whisper backends
    whisper_check = WhisperLocalProvider()
    openai_whisper_available = whisper_check._check_openai_whisper()
    faster_whisper_available = whisper_check._check_faster_whisper()

    # Determine smart defaults based on availability
    default_llm = "claude" if anthropic_key_set else ("ollama" if ollama_available else "claude")
    # Prefer openai-whisper if available (reuses cached models), otherwise auto
    default_backend = "openai-whisper" if openai_whisper_available else ("faster-whisper" if faster_whisper_available else "auto")

    # Provider settings (will be set by prompts or defaults)
    transcription_provider = "whisper_local"
    whisper_backend = default_backend
    whisper_model = "medium"
    llm_provider = default_llm
    llm_model = None

    if not non_interactive:
        console.print("\n[bold]Provider Configuration[/bold]")
        console.print("[dim]Press Enter to accept defaults shown in brackets.[/dim]\n")

        # Show availability status
        console.print("[bold]Available providers:[/bold]")
        console.print("  Transcription:")
        console.print(f"    - whisper_local [dim]- Free, runs locally[/dim]")
        console.print(f"      Backends:")
        console.print(f"        - openai-whisper [dim]- {'[green]Installed[/green]' if openai_whisper_available else '[yellow]Not installed[/yellow]'} (uses standard model cache)[/dim]")
        console.print(f"        - faster-whisper [dim]- {'[green]Installed[/green]' if faster_whisper_available else '[yellow]Not installed[/yellow]'} (faster, separate cache)[/dim]")
        console.print(f"    - whisper_api (OpenAI) [dim]- {'[green]API key set[/green]' if openai_key_set else '[yellow]Requires OPENAI_API_KEY[/yellow]'}[/dim]")
        console.print("  LLM Analysis:")
        console.print(f"    - ollama [dim]- {'[green]Running[/green]' if ollama_available else '[yellow]Not running[/yellow]'} (free, local)[/dim]")
        console.print(f"    - claude [dim]- {'[green]API key set[/green]' if anthropic_key_set else '[yellow]Requires ANTHROPIC_API_KEY[/yellow]'}[/dim]")
        console.print(f"    - openai [dim]- {'[green]API key set[/green]' if openai_key_set else '[yellow]Requires OPENAI_API_KEY[/yellow]'}[/dim]")
        console.print()

        # Prompt for transcription provider
        transcription_provider = typer.prompt(
            "Transcription provider (whisper_local/whisper_api)",
            default="whisper_local",
        ).strip().lower()
        if transcription_provider not in ("whisper_local", "whisper_api"):
            console.print(f"[yellow]Unknown provider '{transcription_provider}', using whisper_local[/yellow]")
            transcription_provider = "whisper_local"

        # Prompt for whisper backend (only if using local)
        if transcription_provider == "whisper_local":
            whisper_backend = typer.prompt(
                "Whisper backend (auto/openai-whisper/faster-whisper)",
                default=default_backend,
            ).strip().lower()
            if whisper_backend not in ("auto", "openai-whisper", "faster-whisper"):
                console.print(f"[yellow]Unknown backend '{whisper_backend}', using auto[/yellow]")
                whisper_backend = "auto"

        # Prompt for whisper model
        whisper_model = typer.prompt(
            "Whisper model (tiny/base/small/medium/large/large-v2/large-v3)",
            default="medium",
        ).strip().lower()
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if whisper_model not in valid_models:
            console.print(f"[yellow]Unknown model '{whisper_model}', using medium[/yellow]")
            whisper_model = "medium"

        # Prompt for LLM provider
        llm_provider = typer.prompt(
            "LLM provider for highlights (claude/openai/ollama)",
            default=default_llm,
        ).strip().lower()
        if llm_provider not in ("claude", "openai", "ollama"):
            console.print(f"[yellow]Unknown provider '{llm_provider}', using {default_llm}[/yellow]")
            llm_provider = default_llm

        # Show warning if selected provider might not work
        if llm_provider == "claude" and not anthropic_key_set:
            console.print("[yellow]Note: ANTHROPIC_API_KEY not set. Set it before running highlights.[/yellow]")
        elif llm_provider == "openai" and not openai_key_set:
            console.print("[yellow]Note: OPENAI_API_KEY not set. Set it before running highlights.[/yellow]")
        elif llm_provider == "ollama" and not ollama_available:
            console.print("[yellow]Note: Ollama not running. Start it with 'ollama serve' before running highlights.[/yellow]")

        console.print()

    # Create directory structure
    directories = [
        brand_path / "videos",
        brand_path / "transcripts",
        brand_path / "projects",
        brand_path / "outputs",
        brand_path / "search_results",
        brand_path / "logo",  # For brand logo files
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Import settings classes
    from clip_video.config import PortraitSettings, LogoSettings, SocialCopyStyle

    # Create comprehensive default configuration with placeholders
    config = BrandConfig(
        name=brand_name,
        description=description,
        # Portrait conversion settings
        portrait=PortraitSettings(
            crop_x_offset=0.5,  # 0.0=left, 0.5=center, 1.0=right
            crop_x_pixels=None,  # Or set exact pixel offset for specific source resolution
            crop_source_width=None,  # Reference width when using pixel offset (e.g., 1920)
        ),
        # Logo overlay settings
        logo=LogoSettings(
            enabled=False,  # Set to true and add logo file to enable
            image_path="logo/logo.png",  # Path relative to brand folder
            position="top-center",  # top-left, top-center, top-right, bottom-left, bottom-center, bottom-right
            height_percent=0.15,  # Logo height as percentage of video height
            opacity=1.0,
            margin=20,
        ),
        # Social media copy style
        social_copy=SocialCopyStyle(
            enabled=True,
            locale="american",  # "american" or "british"
            tone="informative",  # informative, casual, enthusiastic, professional
            voice_description="",  # Describe your brand voice, e.g., "Professional but approachable tech educator"
            avoid_phrases=[
                "game-changer",
                "crushing it",
                "let that sink in",
                "here's the thing",
                "but here's the kicker",
                "I'll be honest",
                "hot take",
                "unpopular opinion",
                "this is huge",
                "mind = blown",
            ],
            preferred_phrases=[],  # Add phrases you want to encourage
            include_hashtags=True,
            default_hashtags=[],  # e.g., ["#YourBrand", "#TechConf2024"]
            max_hook_length=100,
            max_description_length=280,
            custom_prompt="",  # Additional instructions for copy generation
        ),
        # Caption styling
        caption_font="Arial",
        caption_size=48,
        caption_color="#FFFFFF",
        caption_bg_color="#000000",
        caption_bg_opacity=0.7,
        # Triggers (word -> emoji or logo path)
        emoji_triggers={},  # e.g., {"kubernetes": "☸️", "docker": "🐳"}
        logo_triggers={},  # e.g., {"aws": "logos/aws.png"}
        # Vocabulary: maps correct word to common Whisper mistranscriptions
        # The tool will search for both the correct word AND all alternatives
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
        # API providers (configured during init)
        transcription_provider=transcription_provider,
        whisper_backend=whisper_backend,
        whisper_model=whisper_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    config_path = save_brand_config(brand_name, config)

    # Display success message
    llm_display = llm_provider.capitalize() if llm_provider != "ollama" else "Ollama (local)"
    backend_display = whisper_backend if transcription_provider == "whisper_local" else "N/A"
    console.print(
        Panel(
            f"[green]Brand '{brand_name}' created successfully![/green]\n\n"
            f"Location: {brand_path}\n\n"
            f"[bold]Configured providers:[/bold]\n"
            f"  Transcription: {transcription_provider} (backend: {backend_display}, model: {whisper_model})\n"
            f"  LLM Analysis:  {llm_display}\n\n"
            "Directory structure:\n"
            f"  {brand_path}/\n"
            "    videos/         - Place source videos here\n"
            "    transcripts/    - Generated transcripts\n"
            "    projects/       - Lyric match projects\n"
            "    highlights/     - Highlight extraction projects\n"
            "    outputs/        - Generated clips\n"
            "    search_results/ - Search result clips\n"
            "    logo/           - Brand logo files\n"
            f"    config.json     - Brand configuration (edit to customize)\n\n"
            "Next steps:\n"
            "  1. Copy video files to the videos/ directory\n"
            "  2. Run: clip-video transcribe " + brand_name + "\n"
            "  3. Edit config.json to customize voice, vocabulary, logo, etc.",
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
    backend: Annotated[
        Optional[str],
        typer.Option("--backend", "-b", help="Whisper backend (auto, openai-whisper, faster-whisper)"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)"),
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

    # Determine provider, backend, and model
    provider_name = provider or config.transcription_provider
    backend_name = backend or getattr(config, 'whisper_backend', 'auto')
    model_name = model or getattr(config, 'whisper_model', 'medium')

    # Initialize provider with fallback logic
    local_provider = WhisperLocalProvider(model=model_name, backend=backend_name)
    api_provider = WhisperAPIProvider()

    if provider_name == "whisper_local":
        if local_provider.is_available():
            transcription_provider = local_provider
            # Show which backend will be used
            active_backend = local_provider.get_active_backend()
            console.print(f"[dim]Using local Whisper backend: {active_backend}[/dim]")
        elif api_provider.is_available():
            console.print("[yellow]Warning:[/yellow] Local Whisper not available, falling back to API.")
            console.print("[dim]Install a local backend for free transcription:[/dim]")
            console.print("[dim]  pip install openai-whisper  (uses standard model cache)[/dim]")
            console.print("[dim]  pip install faster-whisper  (faster, separate cache)[/dim]")
            transcription_provider = api_provider
            provider_name = "whisper_api"
        else:
            console.print("[red]Error:[/red] No transcription provider available.")
            console.print("Install a local Whisper backend:")
            console.print("  pip install openai-whisper  (recommended if you have models cached)")
            console.print("  pip install faster-whisper  (faster, but downloads separate models)")
            console.print("Or set OPENAI_API_KEY for API transcription.")
            raise typer.Exit(1)
    elif provider_name == "whisper_api":
        if api_provider.is_available():
            transcription_provider = api_provider
        elif local_provider.is_available():
            console.print("[yellow]Warning:[/yellow] OpenAI API not configured, falling back to local Whisper.")
            transcription_provider = local_provider
            provider_name = "whisper_local"
        else:
            console.print("[red]Error:[/red] No transcription provider available.")
            console.print("Set OPENAI_API_KEY for API transcription.")
            console.print("Or install a local Whisper backend:")
            console.print("  pip install openai-whisper")
            console.print("  pip install faster-whisper")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Error:[/red] Unknown provider '{provider_name}'.")
        console.print("Use 'whisper_local' or 'whisper_api'.")
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

    from clip_video.search import BrandSearcher
    from clip_video.ffmpeg import FFmpegWrapper, ExtractionConfig

    brand_path = get_brand_path(brand_name)

    # Check for transcripts
    transcripts_dir = brand_path / "transcripts"
    transcript_count = len(list(transcripts_dir.glob("*.json"))) if transcripts_dir.exists() else 0

    if transcript_count == 0:
        console.print(f"[red]Error:[/red] No transcripts found for brand '{brand_name}'.")
        console.print(f"Run: clip-video transcribe {brand_name}")
        raise typer.Exit(1)

    console.print(f"[cyan]Searching {transcript_count} transcripts for:[/cyan] '{phrase}'")

    # Initialize searcher
    searcher = BrandSearcher(brand_name)

    # Perform search
    results = searcher.search(phrase, max_results=limit)

    if not results.results:
        console.print(f"\n[yellow]No matches found for '{phrase}'[/yellow]")
        return

    # Display results
    table = Table(title=f"Search Results for '{phrase}' ({len(results.results)} matches)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Video", style="cyan", max_width=30)
    table.add_column("Time", style="white")
    table.add_column("Context", style="dim", max_width=50)
    table.add_column("Score", style="green", justify="right")

    for i, result in enumerate(results.results[:limit], 1):
        video_name = Path(result.video_id).stem[:28]
        time_range = f"{result.start:.1f}s - {result.end:.1f}s"
        context = f"...{result.context_before[-20:]} [{result.phrase}] {result.context_after[:20]}..."
        score = f"{result.confidence:.0%}"
        table.add_row(str(i), video_name, time_range, context, score)

    console.print()
    console.print(table)

    # Export clips if requested
    if export:
        console.print(f"\n[cyan]Exporting clips...[/cyan]")
        output_dir = brand_path / "search_results" / phrase.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg = FFmpegWrapper()
        videos_dir = brand_path / "videos"
        exported = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress_bar:
            task = progress_bar.add_task("Exporting...", total=min(limit, len(results.results)))

            for i, result in enumerate(results.results[:limit], 1):
                # Find video file
                video_path = videos_dir / f"{result.video_id}.mp4"
                if not video_path.exists():
                    # Try other extensions
                    for ext in [".mkv", ".mov", ".avi", ".webm"]:
                        alt_path = videos_dir / f"{result.video_id}{ext}"
                        if alt_path.exists():
                            video_path = alt_path
                            break

                if not video_path.exists():
                    progress_bar.advance(task)
                    continue

                # Extract clip
                output_path = output_dir / f"{phrase.replace(' ', '_')}_{i:02d}.mp4"
                try:
                    ffmpeg.extract_clip(
                        input_path=video_path,
                        output_path=output_path,
                        start_time=result.start,
                        end_time=result.end,
                    )
                    exported += 1
                except Exception as e:
                    console.print(f"\n[yellow]Warning:[/yellow] Failed to export clip {i}: {e}")

                progress_bar.advance(task)

        console.print(f"\n[green]Exported {exported} clips to:[/green] {output_dir}")


@app.command()
def index_transcripts(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand to index")],
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", "-r", help="Rebuild index from scratch"),
    ] = False,
) -> None:
    """Build or rebuild the transcript search index for a brand.

    Creates an inverted index of all words in the brand's transcripts,
    enabling fast word/phrase search across the video library.

    This must be run before using the search or lyric-match commands.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    from clip_video.transcription import TranscriptionResult
    from clip_video.transcript.index import TranscriptIndex, TranscriptIndexManager
    from clip_video.models.transcript import Transcript, TranscriptSegment, TranscriptWord

    brand_path = get_brand_path(brand_name)
    transcripts_dir = brand_path / "transcripts"

    # Find all transcript files
    transcript_files = list(transcripts_dir.glob("*.json")) if transcripts_dir.exists() else []
    # Exclude progress file
    transcript_files = [f for f in transcript_files if not f.name.startswith(".")]

    if not transcript_files:
        console.print(f"[red]Error:[/red] No transcripts found for brand '{brand_name}'.")
        console.print(f"Run: clip-video transcribe {brand_name}")
        raise typer.Exit(1)

    # Check if index exists
    index_manager = TranscriptIndexManager(brand_path.parent)
    index_path = brand_path / "transcript_index.json"

    if index_path.exists() and not rebuild:
        console.print(f"[yellow]Index already exists.[/yellow] Use --rebuild to recreate.")
        # Show index stats
        index = index_manager.get(brand_name)
        stats = index.get_statistics()
        console.print(f"\n[cyan]Current index:[/cyan]")
        console.print(f"  • Transcripts indexed: {stats['indexed_transcripts']}")
        console.print(f"  • Unique words: {stats['unique_words']:,}")
        console.print(f"  • Total word occurrences: {stats['total_occurrences']:,}")
        return

    console.print(f"[cyan]Building transcript index for {brand_name}...[/cyan]")
    console.print(f"Found {len(transcript_files)} transcript files\n")

    # Create new index
    index = TranscriptIndex(brand_name=brand_name)
    total_words = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Indexing...", total=len(transcript_files))

        for transcript_file in transcript_files:
            video_id = transcript_file.stem

            try:
                # Load the transcript in the old format
                old_transcript = TranscriptionResult.load(transcript_file)

                # Convert to the new Transcript model format
                segments = []
                for seg_idx, seg in enumerate(old_transcript.segments):
                    words = [
                        TranscriptWord(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            confidence=w.confidence,
                        )
                        for w in seg.words
                    ]
                    segment = TranscriptSegment(
                        id=seg_idx,
                        text=seg.text,
                        start=seg.start,
                        end=seg.end,
                        words=words,
                    )
                    segments.append(segment)

                transcript = Transcript(
                    video_path=old_transcript.video_path,
                    language=old_transcript.language,
                    provider=old_transcript.provider,
                    model=old_transcript.model,
                    duration=old_transcript.duration,
                    segments=segments,
                    full_text=old_transcript.text,
                )

                # Add to index (use brand_name as project_name for legacy transcripts)
                word_count = index.add_transcript(brand_name, video_id, transcript)
                total_words += word_count

            except Exception as e:
                console.print(f"\n[yellow]Warning:[/yellow] Failed to index {video_id}: {e}")

            progress_bar.advance(task)

    # Save the index
    index.save(index_path)

    # Show stats
    stats = index.get_statistics()
    console.print(f"\n[green]Index built successfully![/green]")
    console.print(f"\n[cyan]Index statistics:[/cyan]")
    console.print(f"  • Transcripts indexed: {stats['indexed_transcripts']}")
    console.print(f"  • Unique words: {stats['unique_words']:,}")
    console.print(f"  • Total word occurrences: {stats['total_occurrences']:,}")
    console.print(f"\n[dim]Index saved to: {index_path}[/dim]")


@app.command()
def export_dictionary(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (default: brands/{brand}/dictionary.txt)"),
    ] = None,
    min_occurrences: Annotated[
        int,
        typer.Option("--min", "-m", help="Minimum occurrences to include a word"),
    ] = 1,
    sort_by: Annotated[
        str,
        typer.Option("--sort", "-s", help="Sort by: 'alpha' (alphabetical) or 'count' (frequency)"),
    ] = "alpha",
    include_counts: Annotated[
        bool,
        typer.Option("--counts", "-c", help="Include occurrence counts in output"),
    ] = False,
    include_numbers: Annotated[
        bool,
        typer.Option("--numbers", "-n", help="Include purely numeric words (default: exclude)"),
    ] = False,
):
    """Export a dictionary of all unique words found across a brand's transcripts.

    This is useful for adapting song lyrics to match available words in your video library.
    """
    from clip_video.search import TranscriptIndex

    brands_root = Path("brands")
    brand_dir = brands_root / brand_name

    if not brand_dir.exists():
        console.print(f"[red]Error:[/red] Brand '{brand_name}' not found")
        raise typer.Exit(1)

    # Load the transcript index
    index_path = brand_dir / "transcript_index.json"
    if not index_path.exists():
        console.print(f"[yellow]No transcript index found.[/yellow]")
        console.print(f"Run [cyan]clip-video index-transcripts {brand_name}[/cyan] first.")
        raise typer.Exit(1)

    index = TranscriptIndex.load(index_path)
    stats = index.get_statistics()

    if stats["unique_words"] == 0:
        console.print("[yellow]No words found in the index.[/yellow]")
        raise typer.Exit(1)

    # Get all words with their counts
    word_counts: dict[str, int] = {}
    for word, occurrences in index.words.items():
        # Skip purely numeric words unless explicitly requested
        if not include_numbers and word.isdigit():
            continue
        count = len(occurrences)
        if count >= min_occurrences:
            word_counts[word] = count

    # Sort
    if sort_by == "count":
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    else:  # alpha
        sorted_words = sorted(word_counts.items(), key=lambda x: x[0])

    # Determine output path
    if output is None:
        output = brand_dir / "dictionary.txt"

    # Write dictionary
    with open(output, "w", encoding="utf-8") as f:
        if include_counts:
            for word, count in sorted_words:
                f.write(f"{word}\t{count}\n")
        else:
            for word, _ in sorted_words:
                f.write(f"{word}\n")

    # Show summary
    console.print(f"\n[green]Dictionary exported successfully![/green]")
    console.print(f"\n[cyan]Summary:[/cyan]")
    console.print(f"  • Words exported: {len(sorted_words):,}")
    console.print(f"  • Min occurrences filter: {min_occurrences}")
    console.print(f"  • Sorted by: {sort_by}")
    console.print(f"\n[dim]Saved to: {output}[/dim]")

    # Show sample
    console.print(f"\n[cyan]Sample words:[/cyan]")
    sample = sorted_words[:20]
    if include_counts:
        for word, count in sample:
            console.print(f"  {word} ({count})")
    else:
        for word, _ in sample:
            console.print(f"  {word}")
    if len(sorted_words) > 20:
        console.print(f"  ... and {len(sorted_words) - 20:,} more")


@app.command()
def check_lyrics(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    lyrics_file: Annotated[Path, typer.Argument(help="Path to lyrics text file")],
    show_found: Annotated[
        bool,
        typer.Option("--found", "-f", help="Also show words that were found"),
    ] = False,
    by_line: Annotated[
        bool,
        typer.Option("--by-line", "-l", help="Show missing words organized by line"),
    ] = False,
):
    """Check which words from a lyrics file are available in the brand's transcripts.

    Quickly identifies missing words so you can adapt lyrics to match available clips.
    """
    from clip_video.search import TranscriptIndex
    from clip_video.lyrics.parser import LyricsParser

    brands_root = Path("brands")
    brand_dir = brands_root / brand_name

    if not brand_dir.exists():
        console.print(f"[red]Error:[/red] Brand '{brand_name}' not found")
        raise typer.Exit(1)

    if not lyrics_file.exists():
        console.print(f"[red]Error:[/red] Lyrics file not found: {lyrics_file}")
        raise typer.Exit(1)

    # Load the transcript index
    index_path = brand_dir / "transcript_index.json"
    if not index_path.exists():
        console.print(f"[yellow]No transcript index found.[/yellow]")
        console.print(f"Run [cyan]clip-video index-transcripts {brand_name}[/cyan] first.")
        raise typer.Exit(1)

    index = TranscriptIndex.load(index_path)
    available_words = set(index.words.keys())

    # Parse lyrics
    parser = LyricsParser()
    lyrics = parser.parse_file(lyrics_file)

    # Analyze each line
    all_found: set[str] = set()
    all_missing: set[str] = set()
    lines_with_missing: list[tuple[int, str, list[str]]] = []

    for line in lyrics.content_lines:
        line_missing = []
        for word in line.words:
            if word in available_words:
                all_found.add(word)
            else:
                all_missing.add(word)
                line_missing.append(word)
        if line_missing:
            lines_with_missing.append((line.line_number, line.raw_text.strip(), line_missing))

    # Calculate coverage
    total_unique = len(all_found | all_missing)
    coverage = (len(all_found) / total_unique * 100) if total_unique > 0 else 100

    # Display results
    console.print(f"\n[cyan]Lyrics Analysis: {lyrics_file.name}[/cyan]")
    console.print(f"[dim]Brand: {brand_name}[/dim]\n")

    # Summary
    console.print(f"[bold]Coverage:[/bold] {coverage:.1f}%")
    console.print(f"  • Words found: [green]{len(all_found)}[/green]")
    console.print(f"  • Words missing: [red]{len(all_missing)}[/red]")

    # Missing words
    if all_missing:
        console.print(f"\n[red]Missing words ({len(all_missing)}):[/red]")
        if by_line:
            for line_num, line_text, missing in lines_with_missing:
                console.print(f"\n  [dim]Line {line_num}:[/dim] {line_text}")
                for word in missing:
                    console.print(f"    [red]x[/red] {word}")
        else:
            for word in sorted(all_missing):
                console.print(f"  [red]x[/red] {word}")
    else:
        console.print(f"\n[green]All words found![/green]")

    # Found words (optional)
    if show_found and all_found:
        console.print(f"\n[green]Found words ({len(all_found)}):[/green]")
        for word in sorted(all_found):
            console.print(f"  [green]>[/green] {word}")


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
        typer.Option("--resume", "-r", help="Resume existing project (uses old lyrics)"),
    ] = False,
    update: Annotated[
        bool,
        typer.Option("--update", "-u", help="Update project with new lyrics (keeps existing clips)"),
    ] = False,
    max_candidates: Annotated[
        int,
        typer.Option("--candidates", "-c", help="Max candidate clips per word/phrase"),
    ] = 5,
    no_words: Annotated[
        bool,
        typer.Option("--no-words", help="Don't extract individual words"),
    ] = False,
    no_phrases: Annotated[
        bool,
        typer.Option("--no-phrases", help="Don't extract phrases"),
    ] = False,
    search_only: Annotated[
        bool,
        typer.Option("--search-only", help="Only search, don't extract clips"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
) -> None:
    """Start a lyric match project.

    Parses the lyrics file and extracts candidate clips for each word/phrase
    from the brand's video library.

    The lyrics file can include:
    - Section markers: [Verse 1], [Chorus], etc.
    - Phrase markers: [phrase]multi word phrase[/phrase]
    - Repeat markers: (x2), (x3)
    - Metadata: Title: Song Name, Artist: Artist Name
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    if resume and update:
        console.print(f"[red]Error:[/red] Cannot use both --resume and --update.")
        raise typer.Exit(1)

    if not resume and not lyrics_file.exists():
        console.print(f"[red]Error:[/red] Lyrics file not found: {lyrics_file}")
        raise typer.Exit(1)

    from clip_video.modes.lyric_match import (
        LyricMatchProcessor,
        LyricMatchConfig,
        LyricMatchProject,
    )

    brand_path = get_brand_path(brand_name)

    # Check for transcripts
    transcripts_dir = brand_path / "transcripts"
    transcript_count = len(list(transcripts_dir.glob("*.json"))) if transcripts_dir.exists() else 0

    if transcript_count == 0:
        console.print(f"[red]Error:[/red] No transcripts found for brand '{brand_name}'.")
        console.print(f"Run: clip-video transcribe {brand_name}")
        raise typer.Exit(1)

    # Create config
    config = LyricMatchConfig(
        max_candidates_per_target=max_candidates,
        extract_words=not no_words,
        extract_phrases=not no_phrases,
    )

    # Initialize processor
    processor = LyricMatchProcessor(brand_name, config=config)

    # Load or create project
    new_targets = []
    removed_targets = []

    if resume:
        project = processor.load_project(project_name)
        if not project:
            console.print(f"[red]Error:[/red] Project '{project_name}' not found.")
            console.print(f"Create a new project by removing --resume flag.")
            raise typer.Exit(1)
        console.print(f"[cyan]Resuming project:[/cyan] {project_name}")
    elif update:
        project = processor.load_project(project_name)
        if not project:
            console.print(f"[red]Error:[/red] Project '{project_name}' not found.")
            console.print(f"Create a new project by removing --update flag.")
            raise typer.Exit(1)
        console.print(f"[cyan]Updating project:[/cyan] {project_name}")
        new_targets, removed_targets = processor.update_project(project, lyrics_file)

        if new_targets or removed_targets:
            console.print(f"  [green]+{len(new_targets)}[/green] new targets")
            console.print(f"  [red]-{len(removed_targets)}[/red] removed targets")
            if new_targets:
                console.print(f"  New: {', '.join(t.text for t in new_targets[:10])}" +
                              (f" (+{len(new_targets)-10} more)" if len(new_targets) > 10 else ""))
        else:
            console.print(f"  [yellow]No changes detected in lyrics[/yellow]")
    else:
        # Check if project exists
        existing = processor.load_project(project_name)
        if existing:
            if not yes:
                if not typer.confirm(f"Project '{project_name}' already exists. Overwrite?"):
                    console.print("[yellow]Cancelled.[/yellow]")
                    raise typer.Exit(0)

        console.print(f"[cyan]Creating project:[/cyan] {project_name}")
        project = processor.create_project(project_name, lyrics_file)

    # Show project summary
    summary = project.get_summary()
    console.print(Panel(
        f"[bold]Lyric Match Project[/bold]\n\n"
        f"Lyrics: {project.lyrics_file.name}\n"
        f"Lines: {summary['total_lines']}\n"
        f"Words: {len(project.extraction_list.unique_words) if project.extraction_list else 0}\n"
        f"Phrases: {len(project.extraction_list.unique_phrases) if project.extraction_list else 0}\n"
        f"Total targets: {summary['total_targets']}\n"
        f"Transcripts to search: {transcript_count}",
        title="Project Summary",
    ))

    if not yes:
        if not typer.confirm("Proceed with search?"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    # Search phase
    console.print()
    console.print("[bold]Phase 1: Searching transcripts...[/bold]")

    search_results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        total_targets = summary["total_targets"]
        task = progress_bar.add_task("Searching...", total=total_targets)

        def search_progress(target: str, current: int, total: int):
            progress_bar.update(task, completed=current, description=f"Searching: {target[:30]}...")

        search_results = processor.search_all(project, progress_callback=search_progress)

    # Count found targets
    found_targets = sum(1 for r in search_results.values() if r.results)
    console.print(f"\n[green]Found matches for {found_targets}/{summary['total_targets']} targets[/green]")

    if search_only:
        # Save report and exit
        report_path = processor.save_report(project)
        console.print(f"\n[cyan]Report saved to:[/cyan] {report_path}")
        return

    # Extraction phase
    console.print()
    console.print("[bold]Phase 2: Extracting clips...[/bold]")

    # Get video paths
    video_paths = processor.get_video_paths(project)
    if not video_paths:
        console.print("[yellow]Warning:[/yellow] No video files found in brand directory.")
        console.print("Add video files to: " + str(brand_path / "videos"))
        report_path = processor.save_report(project)
        console.print(f"\n[cyan]Report saved to:[/cyan] {report_path}")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Extracting...", total=summary["total_targets"])

        def extract_progress(target: str, current: int, total: int):
            progress_bar.update(task, completed=current, description=f"Extracting: {target[:30]}...")

        total_clips = processor.extract_candidates(
            project,
            search_results,
            video_paths,
            progress_callback=extract_progress,
        )

    # Generate report
    report_path = processor.save_report(project)

    # Get word and phrase coverage
    words_found, total_words, missing_words = project.get_word_coverage()
    phrases_found, total_phrases = project.get_phrase_coverage()

    word_coverage_pct = (words_found / total_words * 100) if total_words > 0 else 100.0

    # Show final summary with word coverage as primary metric
    console.print()

    # Word coverage is the critical metric
    if words_found == total_words:
        word_status = f"[bold green]✓ Word coverage: {words_found}/{total_words} (100%)[/bold green]"
    else:
        word_status = f"[bold red]✗ Word coverage: {words_found}/{total_words} ({word_coverage_pct:.0f}%)[/bold red]"

    console.print(Panel(
        f"[bold green]Lyric Match Complete![/bold green]\n\n"
        f"{word_status}\n"
        f"Phrase coverage: {phrases_found}/{total_phrases}\n"
        f"Total clips extracted: {total_clips}\n\n"
        f"Output folder: {project.clips_dir}\n"
        f"Report: {report_path}",
        title="Results",
    ))

    # Show missing words prominently (this is actionable)
    if missing_words:
        console.print(f"\n[bold red]Missing words ({len(missing_words)}):[/bold red]")
        # Show all missing words since these are critical
        for word in missing_words:
            console.print(f"  • \"{word}\"")
        console.print()
        console.print("[dim]Missing words need lyrics changes, more source material, or manual recording.[/dim]")


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
    llm_provider: Annotated[
        Optional[str],
        typer.Option("--llm-provider", "-l", help="LLM provider (claude, openai, ollama)"),
    ] = None,
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", "-m", help="LLM model (e.g., claude-sonnet-4-5, gpt-4.1, llama3.2)"),
    ] = None,
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
    from clip_video.llm import LLMConfig, LLMProviderType

    brand_path = get_brand_path(brand_name)
    config = load_brand_config(brand_name)

    # Get LLM provider with fallback logic
    actual_provider, actual_model = get_llm_provider_with_fallback(
        brand_name, llm_provider, llm_model
    )

    # Check for existing transcript
    transcript_path = brand_path / "transcripts" / f"{video.stem}.json"
    if not transcript_path.exists():
        console.print(f"[red]Error:[/red] No transcript found for {video.name}")
        console.print(f"Run: clip-video transcribe {brand_name} --video {video}")
        raise typer.Exit(1)

    # Load transcript
    transcript_result = TranscriptionResult.load(transcript_path)

    # Create LLM config with selected provider
    provider_type = LLMProviderType(actual_provider)
    llm_config = LLMConfig(provider=provider_type, model=actual_model)

    # Create highlights config
    highlights_config = HighlightsConfig(target_clips=count, llm_config=llm_config)

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

    # Show provider info
    provider_display = actual_provider.capitalize()
    if actual_provider == "ollama":
        provider_display = "Ollama (Local)"
    model_display = actual_model or "(default)"

    console.print(Panel(
        f"[bold]Highlights Generation[/bold]\n\n"
        f"Video: {video.name}\n"
        f"Target clips: {count}\n"
        f"LLM Provider: {provider_display}\n"
        f"LLM Model: {model_display}\n"
        + (f"Estimated LLM cost: ${estimated_cost:.3f} USD" if estimated_cost > 0 else "Estimated LLM cost: Free (local)"),
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

    # Check LLM API key availability
    check_llm_api_key(brand_name)

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
        # Try different encodings (Windows often saves as UTF-16)
        content = None
        for encoding in ["utf-8", "utf-16", "utf-16-le", "latin-1"]:
            try:
                content = video_list.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if content is None:
            console.print(f"[red]Error:[/red] Could not read video list file: {video_list}")
            raise typer.Exit(1)
        lines = content.strip().splitlines()
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


# =============================================================================
# Review Queue Commands
# =============================================================================

# Create review subcommand group
review_app = typer.Typer(
    name="review",
    help="Manage the review queue for rejected clips.",
)
app.add_typer(review_app, name="review")


@review_app.command("list")
def review_list(
    brand_name: Annotated[
        Optional[str],
        typer.Argument(help="Filter by brand name"),
    ] = None,
    video: Annotated[
        Optional[str],
        typer.Option("--video", "-v", help="Filter by video name"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of clips to show"),
    ] = 20,
) -> None:
    """List all clips in the review queue.

    Shows rejected clips that need human review, with their rejection
    reasons and preview commands.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        # Use current working directory
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[yellow]No review queue found.[/yellow]")
        console.print(f"Expected location: {review_dir}")
        return

    queue = ReviewQueue(review_dir)
    clips = queue.list_all()

    # Filter by video if specified
    if video:
        clips = [c for c in clips if video.lower() in c.video_path.lower()]

    if not clips:
        console.print("[green]Review queue is empty.[/green]")
        return

    # Limit results
    clips = clips[:limit]

    # Display table
    table = Table(title=f"Review Queue ({queue.count()} clips)")
    table.add_column("Clip ID", style="cyan", max_width=25)
    table.add_column("Time", style="white")
    table.add_column("Duration", style="green")
    table.add_column("Rejection Reasons", style="yellow", max_width=40)
    table.add_column("Video", style="dim", max_width=30)

    for clip in clips:
        time_range = f"{clip.start_time:.1f}s - {clip.end_time:.1f}s"
        duration = f"{clip.duration:.1f}s"
        reasons = "\n".join(clip.rejection_reasons[:3])
        if len(clip.rejection_reasons) > 3:
            reasons += f"\n... +{len(clip.rejection_reasons) - 3} more"
        video_name = Path(clip.video_path).name[:30]

        table.add_row(clip.clip_id, time_range, duration, reasons, video_name)

    console.print(table)

    # Show summary
    summary = queue.get_summary()
    console.print(f"\n[dim]Total duration: {summary['total_duration']:.1f}s[/dim]")

    if len(clips) < queue.count():
        console.print(f"[dim]Showing {len(clips)} of {queue.count()} clips. Use --limit to show more.[/dim]")


@review_app.command("show")
def review_show(
    clip_id: Annotated[str, typer.Argument(help="Clip ID to show details for")],
    brand_name: Annotated[
        Optional[str],
        typer.Option("--brand", "-b", help="Brand name"),
    ] = None,
) -> None:
    """Show detailed information about a rejected clip.

    Displays full rejection reasons, validation details, transcript
    segment, and preview commands.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[red]Error:[/red] No review queue found.")
        raise typer.Exit(1)

    queue = ReviewQueue(review_dir)
    clip = queue.get(clip_id)

    if not clip:
        console.print(f"[red]Error:[/red] Clip '{clip_id}' not found in review queue.")
        raise typer.Exit(1)

    # Display detailed information
    console.print(Panel(
        f"[bold]Clip ID:[/bold] {clip.clip_id}\n"
        f"[bold]Video:[/bold] {clip.video_path}\n"
        f"[bold]Time Range:[/bold] {clip.start_time:.1f}s - {clip.end_time:.1f}s "
        f"({clip.duration:.1f}s duration)\n"
        f"[bold]Rejected At:[/bold] {clip.rejected_at}\n"
        f"[bold]Replacement Attempts:[/bold] {clip.replacement_attempts}",
        title="Clip Details",
    ))

    # Rejection reasons
    console.print("\n[bold yellow]Rejection Reasons:[/bold yellow]")
    for reason in clip.rejection_reasons:
        console.print(f"  • {reason}")

    # Validation details
    if clip.validation_details:
        console.print("\n[bold]Validation Details:[/bold]")
        criteria = clip.validation_details.get("criteria", [])
        for c in criteria:
            status = "[green]PASS[/green]" if c.get("result") == "pass" else "[red]FAIL[/red]"
            console.print(f"  {c.get('criterion')}: {status} - {c.get('reason', '')}")

    # Transcript segment
    console.print("\n[bold]Transcript Segment:[/bold]")
    console.print(Panel(clip.transcript_segment, border_style="dim"))

    # Preview commands
    console.print("\n[bold]Preview Commands:[/bold]")
    console.print(f"  [cyan]clip-video:[/cyan] {clip.preview_command}")
    console.print(f"  [cyan]ffplay:[/cyan] {clip.ffplay_command}")


@review_app.command("approve")
def review_approve(
    clip_id: Annotated[str, typer.Argument(help="Clip ID to approve")],
    brand_name: Annotated[
        Optional[str],
        typer.Option("--brand", "-b", help="Brand name"),
    ] = None,
    extract: Annotated[
        bool,
        typer.Option("--extract", "-e", help="Extract the clip after approval"),
    ] = False,
) -> None:
    """Approve a rejected clip for manual inclusion.

    Removes the clip from the review queue. Optionally extracts the
    clip using --extract flag.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[red]Error:[/red] No review queue found.")
        raise typer.Exit(1)

    queue = ReviewQueue(review_dir)
    clip = queue.get(clip_id)

    if not clip:
        console.print(f"[red]Error:[/red] Clip '{clip_id}' not found in review queue.")
        raise typer.Exit(1)

    # Remove from queue
    queue.remove(clip_id)
    console.print(f"[green]Approved:[/green] Clip '{clip_id}' removed from review queue.")

    if extract:
        console.print("\n[cyan]Extracting clip...[/cyan]")
        # TODO: Integrate with extraction pipeline
        console.print(f"[yellow]Manual extraction:[/yellow]")
        console.print(f"  {clip.ffplay_command.replace('ffplay', 'ffmpeg').replace('-autoexit', '')} output.mp4")


@review_app.command("reject")
def review_reject(
    clip_id: Annotated[str, typer.Argument(help="Clip ID to permanently reject")],
    brand_name: Annotated[
        Optional[str],
        typer.Option("--brand", "-b", help="Brand name"),
    ] = None,
) -> None:
    """Permanently reject a clip.

    Removes the clip from the review queue without extracting it.
    This indicates the clip should not be used.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[red]Error:[/red] No review queue found.")
        raise typer.Exit(1)

    queue = ReviewQueue(review_dir)
    clip = queue.get(clip_id)

    if not clip:
        console.print(f"[red]Error:[/red] Clip '{clip_id}' not found in review queue.")
        raise typer.Exit(1)

    # Remove from queue
    queue.remove(clip_id)
    console.print(f"[red]Rejected:[/red] Clip '{clip_id}' permanently removed from review queue.")


@review_app.command("clear")
def review_clear(
    brand_name: Annotated[
        Optional[str],
        typer.Option("--brand", "-b", help="Brand name"),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Clear all clips from the review queue.

    Permanently removes all rejected clips from the queue.
    This cannot be undone.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[yellow]No review queue found.[/yellow]")
        return

    queue = ReviewQueue(review_dir)
    count = queue.count()

    if count == 0:
        console.print("[green]Review queue is already empty.[/green]")
        return

    if not yes:
        if not typer.confirm(f"Clear {count} clips from review queue?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    removed = queue.clear()
    console.print(f"[green]Cleared {removed} clips from review queue.[/green]")


@review_app.command("summary")
def review_summary(
    brand_name: Annotated[
        Optional[str],
        typer.Option("--brand", "-b", help="Brand name"),
    ] = None,
) -> None:
    """Show summary statistics for the review queue.

    Displays counts by rejection reason and by video source.
    """
    # Determine review directory
    if brand_name:
        if not brand_exists(brand_name):
            console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
            raise typer.Exit(1)
        review_dir = get_brand_path(brand_name) / "review"
    else:
        review_dir = Path.cwd() / "review"

    if not review_dir.exists():
        console.print("[yellow]No review queue found.[/yellow]")
        return

    queue = ReviewQueue(review_dir)
    summary = queue.get_summary()

    if summary["total_clips"] == 0:
        console.print("[green]Review queue is empty.[/green]")
        return

    console.print(Panel(
        f"[bold]Total Clips:[/bold] {summary['total_clips']}\n"
        f"[bold]Total Duration:[/bold] {summary['total_duration']:.1f}s "
        f"({summary['total_duration'] / 60:.1f} minutes)",
        title="Review Queue Summary",
    ))

    # By rejection reason
    if summary["by_reason"]:
        console.print("\n[bold]By Rejection Reason:[/bold]")
        reason_table = Table(show_header=True, header_style="bold")
        reason_table.add_column("Reason", style="yellow")
        reason_table.add_column("Count", style="cyan", justify="right")

        for reason, count in sorted(summary["by_reason"].items(), key=lambda x: -x[1]):
            reason_table.add_row(reason, str(count))

        console.print(reason_table)

    # By video
    if summary["by_video"]:
        console.print("\n[bold]By Source Video:[/bold]")
        video_table = Table(show_header=True, header_style="bold")
        video_table.add_column("Video", style="dim")
        video_table.add_column("Count", style="cyan", justify="right")

        for video, count in sorted(summary["by_video"].items(), key=lambda x: -x[1]):
            video_table.add_row(video[:50], str(count))

        console.print(video_table)


if __name__ == "__main__":
    app()
