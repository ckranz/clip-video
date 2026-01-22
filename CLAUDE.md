# CLAUDE.md - Development Guide for clip-video

## Project Overview

This is a CLI tool for extracting video clips with two primary modes:

1. **Highlights Mode**: AI-powered highlight detection from conference/podcast videos, generating portrait-format social media shorts with burned-in captions

2. **Lyric Match Mode**: Building searchable word/phrase dictionaries from video transcripts for creating music video mashups (speakers "singing" parody songs)

## Quick Reference

```bash
# Run the CLI
clip-video --help

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Architecture

### Directory Structure

```
clip-video/
├── src/clip_video/
│   ├── __init__.py          # Package version
│   ├── cli.py               # Main CLI (Typer-based)
│   ├── config.py            # Brand configuration management
│   ├── ffmpeg.py            # FFmpeg wrapper for video operations
│   ├── ffmpeg_binary.py     # FFmpeg auto-download and path management
│   ├── storage.py           # Atomic JSON read/write utilities
│   ├── transcription/       # Whisper transcription (local & API providers)
│   │
│   ├── modes/
│   │   ├── highlights.py    # Highlights mode processor
│   │   └── lyric_match.py   # Lyric match mode processor
│   │
│   ├── transcript/
│   │   └── index.py         # Transcript search indexing (word/phrase lookup)
│   │
│   ├── lyrics/
│   │   ├── parser.py        # Lyrics file parser (sections, phrases, repeats)
│   │   └── phrases.py       # Phrase extraction and subphrase generation
│   │
│   ├── search.py            # Brand-wide search across transcripts
│   ├── selection.py         # Clip selection tracking
│   ├── captions.py          # Caption rendering for videos
│   └── review/
│       └── queue.py         # Rejected clip review management
│
├── brands/                   # User data directory (gitignored except example)
│   └── example/
│       └── config.json      # Example brand configuration
│
├── tests/                   # Test suite
└── pyproject.toml          # Project configuration
```

### Key Concepts

**Brand**: A collection of videos from a single source (conference, podcast series). Each brand has:
- `videos/` - Source video files
- `transcripts/` - Generated Whisper transcripts with word-level timestamps
- `projects/` - Lyric match projects
- `highlights/` - Highlight extraction projects
- `config.json` - Brand-specific settings (vocabulary, crop settings, logo)

**Transcript Index**: A search index built from all transcripts in a brand. Supports:
- Word search (with alternatives from vocabulary config)
- Phrase search (words appearing consecutively within time gap)
- Returns timestamps for clip extraction

**ExtractionList**: For lyric match, the parsed lyrics broken into:
- Individual words (critical - need 100% coverage)
- Phrases (sliding window subphrases, nice-to-have)

### Data Flow

**Highlights Mode:**
```
Video → Transcribe (Whisper local/API) → Analyze (Claude) → Extract clips → Crop to portrait → Burn captions
```

**Lyric Match Mode:**
```
Lyrics file → Parse → Extract targets → Search transcripts → Extract clips per word/phrase
```

## Key Files to Understand

### CLI (`src/clip_video/cli.py`)
- Typer-based CLI with commands for both modes
- Entry point for all user interactions
- Handles progress display with Rich

### Transcript Index (`src/clip_video/transcript/index.py`)
- `TranscriptIndex`: In-memory index of all words with timestamps
- `search_word()`: Find word occurrences (checks vocabulary alternatives)
- `search_phrase()`: Find consecutive words within max_gap seconds
- `TranscriptIndexManager`: Lazy loading and caching of indices

### Lyric Match Processor (`src/clip_video/modes/lyric_match.py`)
- `LyricMatchProject`: Project state (targets, clips, coverage)
- `LyricMatchProcessor`: Orchestrates search and extraction
- `get_word_coverage()`: Returns (found, total, missing_words) - critical metric
- `get_phrase_coverage()`: Returns (found, total) - nice-to-have metric

### Search (`src/clip_video/search.py`)
- `BrandSearcher`: High-level search across all brand transcripts
- `SearchResult`: Single match with timestamps, confidence, ranking
- `SearchResults`: Collection with grouping and filtering

### Lyrics Parser (`src/clip_video/lyrics/parser.py`, `phrases.py`)
- Parses lyrics files with section markers, phrase markers, repeats
- `PhraseExtractor`: Generates sliding window subphrases
- Deduplicates across repeated lines

## Common Tasks

### Adding a new CLI command

1. Add function in `cli.py` with `@app.command()` decorator
2. Use type hints with `Annotated` for arguments/options
3. Use `console.print()` with Rich markup for output

### Modifying search behavior

The search pipeline:
1. `BrandSearcher.search()` calls `TranscriptIndex.search_phrase()`
2. Results converted to `SearchResult` objects
3. Ranking score calculated based on confidence, duration, exact match

### Modifying clip extraction

1. `FFmpegWrapper.extract_clip()` handles actual extraction
2. `ExtractionConfig` controls padding, format
3. Clips are extracted to flat structure: `clips/{word_or_phrase}/`

### Working with brand config

```python
from clip_video.config import load_brand_config, save_brand_config

config = load_brand_config("my_brand")
config.vocabulary["kubernetes"] = ["cooper netties"]
save_brand_config("my_brand", config)
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_search.py

# Run with coverage
uv run pytest --cov=clip_video
```

## Important Implementation Details

### Zero-duration word handling
Whisper sometimes returns words with identical start/end times. `SearchResult.from_phrase_match()` enforces a minimum duration of 0.5 seconds.

### Idempotent clip extraction
Clips are skipped if output file already exists. Use `--update` flag to re-parse lyrics while keeping existing clips.

### Vocabulary alternatives
The transcript index searches for both the canonical word and all alternatives defined in `config.vocabulary`. This catches common Whisper mistranscriptions.

### Word vs phrase coverage
- **Word coverage**: Critical - need clips for every unique word
- **Phrase coverage**: Nice-to-have - phrases can be assembled from words

The CLI output prioritizes word coverage with red/green status indicators.

## Environment Variables

- `OPENAI_API_KEY`: Optional - for Whisper API transcription or OpenAI LLM
- `ANTHROPIC_API_KEY`: Optional - for Claude LLM (highlight detection)

Can be set in `.env` file in project root.

Note: Both transcription and LLM analysis can run fully locally with no API keys required.

## Transcription

By default, transcription uses local Whisper via `faster-whisper` (free, no API costs).
Install with: `pip install faster-whisper`

Configuration options in brand config:
- `transcription_provider`: `"whisper_local"` (default) or `"whisper_api"`
- `whisper_model`: `"tiny"`, `"base"`, `"small"`, `"medium"` (default), `"large"`, `"large-v2"`, `"large-v3"`

CLI overrides: `--provider` and `--model` flags on the `transcribe` command.

## LLM Analysis (Highlights Mode)

LLM providers for highlight detection and analysis:
- `claude` - Anthropic Claude (default, requires ANTHROPIC_API_KEY)
- `openai` - OpenAI GPT (requires OPENAI_API_KEY)
- `ollama` - Local Ollama (free, requires Ollama running)

Configuration options in brand config:
- `llm_provider`: `"claude"` (default), `"openai"`, or `"ollama"`
- `llm_model`: Model to use (null = provider default)

CLI overrides: `--llm-provider` and `--llm-model` flags on the `highlights` command.

To use Ollama (free local inference):
1. Install Ollama: https://ollama.ai
2. Start server: `ollama serve`
3. Pull a model: `ollama pull llama3.2`
4. Run highlights: `clip-video highlights BRAND VIDEO --llm-provider ollama`

## Common Issues

### "End time must be after start time"
Fixed by minimum duration enforcement in `search.py`. If you see this, the fix may not be installed - run `pip install -e .`

### Missing words in lyric match
Check `coverage_report.md`. Options:
- Add more source videos
- Modify lyrics to use available words
- Add vocabulary alternatives for mistranscribed words

### FFmpeg not found
Run `clip-video check-deps`. FFmpeg auto-downloads on first video operation.

## Code Style

- Type hints throughout
- Dataclasses for data structures
- Rich for CLI output
- Atomic file writes via `storage.py`
- No emojis in code/output unless user requests
