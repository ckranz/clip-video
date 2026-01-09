"""Command-line interface for clip-video.

Uses Typer for a modern, type-hinted CLI experience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
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
) -> None:
    """Transcribe videos for a brand.

    Processes all untranscribed videos in the brand's videos folder,
    or a specific video if --video is provided.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    # TODO: Implement transcription logic in task 7
    console.print(f"[yellow]Transcription not yet implemented.[/yellow]")
    console.print(f"Brand: {brand_name}")
    if video:
        console.print(f"Video: {video}")
    if provider:
        console.print(f"Provider: {provider}")


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

    # TODO: Implement highlights logic in task 16
    console.print(f"[yellow]Highlights mode not yet implemented.[/yellow]")
    console.print(f"Brand: {brand_name}")
    console.print(f"Video: {video}")
    if description:
        console.print(f"Description: {description}")
    console.print(f"Count: {count}")


@app.command()
def highlights_batch(
    brand_name: Annotated[str, typer.Argument(help="Name of the brand")],
    video_list: Annotated[Path, typer.Argument(help="Path to file listing videos to process")],
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of highlights per video"),
    ] = 5,
) -> None:
    """Batch process multiple videos for highlights.

    Reads a list of video files and processes each one to generate
    highlight clips. Progress is saved for resume capability.
    """
    if not brand_exists(brand_name):
        console.print(f"[red]Error:[/red] Brand '{brand_name}' does not exist.")
        raise typer.Exit(1)

    if not video_list.exists():
        console.print(f"[red]Error:[/red] Video list file not found: {video_list}")
        raise typer.Exit(1)

    # TODO: Implement batch processing in task 17
    console.print(f"[yellow]Batch highlights not yet implemented.[/yellow]")
    console.print(f"Brand: {brand_name}")
    console.print(f"Video list: {video_list}")


# =============================================================================
# Utility Commands
# =============================================================================


@app.command()
def check_deps() -> None:
    """Check and report on required dependencies.

    Verifies FFmpeg is available and reports version information.
    """
    # TODO: Implement dependency check in task 2
    console.print(f"[yellow]Dependency check not yet implemented.[/yellow]")
    console.print("Will verify: FFmpeg binaries")


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
