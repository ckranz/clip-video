# SPEC: Clip Video - Intelligent Video Clip Extraction Tool

**Generated from:** DISCOVERY.md
**Date:** 2026-01-09

---

## Goal

Build a Python CLI tool for intelligent video clip extraction with two primary modes:

1. **Lyric Match Mode**: Build searchable word/phrase dictionaries from conference video libraries, enabling rapid creation of music video mashups where speakers "sing" songs
2. **Highlights Mode**: Batch-process conference recordings to automatically generate social media shorts with burned-in captions and brand-specific visual enhancements

Target: Single user processing 15-20 hours of conference content per event on Windows 11 (Surface Book Pro, 32GB RAM).

---

## Tasks

### 1. Project scaffolding and CLI structure
**Depends on:** none
**Description:** Initialize Python project with proper structure. Set up CLI framework using Click or Typer. Create entry points for the two modes (lyric-match, highlights) plus utility commands (init-brand, transcribe, search). Establish configuration loading from brand/project JSON files.
**Acceptance:**
- `pip install -e .` installs the tool
- `clip-video --help` shows available commands
- `clip-video init-brand "BrandName"` creates brand folder structure
- Project follows standard Python packaging (pyproject.toml)
**Files:**
- `pyproject.toml`
- `src/clip_video/__init__.py`
- `src/clip_video/cli.py`
- `src/clip_video/config.py`

### 2. Embedded FFmpeg binary management
**Depends on:** 1
**Description:** Bundle FFmpeg executables directly within the project to eliminate external dependency issues. Use imageio-ffmpeg or ffmpeg-python with static builds, or download platform-specific binaries on first run. Store binaries in a known location within the package. Provide a CLI command to verify/update the FFmpeg installation.
**Acceptance:**
- FFmpeg binaries included or auto-downloaded on first use
- No system PATH or external FFmpeg installation required
- `clip-video check-deps` verifies FFmpeg is available and reports version
- Works on Windows 11 without admin privileges
- Binary location configurable for advanced users
**Files:**
- `src/clip_video/ffmpeg_binary.py`
- `src/clip_video/bin/` (binary storage location)

### 3. Data models and storage layer
**Depends on:** 1
**Description:** Define Pydantic models for Brand, Project, Transcript, and Clip entities. Implement JSON-based persistence with atomic writes (write to temp, rename) to prevent corruption. Create BrandManager and ProjectManager classes for CRUD operations.
**Acceptance:**
- Brand config can be created, loaded, and saved
- Project state persists across CLI invocations
- Transcript data stored with word-level timestamps
- No data corruption on interrupted writes (test with kill -9)
**Files:**
- `src/clip_video/models/brand.py`
- `src/clip_video/models/project.py`
- `src/clip_video/models/transcript.py`
- `src/clip_video/storage.py`

### 4. FFmpeg wrapper for clip extraction
**Depends on:** 2
**Description:** Create FFmpegWrapper class that handles video clip extraction with configurable start/end times and padding. Support both copy mode (fast, no re-encode) and re-encode mode (for format conversion). Include portrait crop functionality with configurable crop region. Output to MP4 container with H.264 video and AAC audio for DaVinci Resolve compatibility. Uses embedded FFmpeg binaries from task 2.
**Acceptance:**
- Extract clip from video with timestamp range
- Apply configurable padding (default: 0.3s before, 0.5s after)
- Convert landscape to portrait with fixed crop
- Output plays correctly in DaVinci Resolve
- Graceful error handling for invalid video files
**Files:**
- `src/clip_video/ffmpeg.py`
- `tests/test_ffmpeg.py`

### 5. Transcript storage and indexing
**Depends on:** 3
**Description:** Design transcript storage format as JSON with word-level timestamps. Build inverted index for fast word/phrase lookup across all transcripts in a brand. Support importing existing VTT/SRT files. Enable export to SRT/VTT formats.
**Acceptance:**
- Store transcript with word timestamps in JSON format
- Search for word returns all occurrences with timestamps and source video
- Phrase search (multi-word) finds consecutive word matches
- Import VTT/SRT preserves timing information
- Index rebuilds correctly after transcript additions
**Files:**
- `src/clip_video/transcript/storage.py`
- `src/clip_video/transcript/index.py`
- `src/clip_video/transcript/import_export.py`
- `tests/test_transcript.py`

### 6. Brand vocabulary and transcript correction
**Depends on:** 3
**Description:** Implement domain-specific vocabulary handling to fix common Whisper transcription errors for technical terms (e.g., "Kubernetes" mis-heard as "Cooper Netties", "ArgoCD" as "Argo seedy"). Brand config includes vocabulary list with canonical terms and known mis-transcriptions. Apply fuzzy matching (Soundex/Metaphone + Levenshtein) to correct transcripts post-processing. Generate Whisper prompt conditioning string from vocabulary to improve initial transcription accuracy.
**Acceptance:**
- Brand config supports `vocabulary` section with terms and known variants
- `clip-video init-brand` includes starter CNCF vocabulary template
- Post-transcription correction pass fixes known mis-transcriptions
- Whisper API calls include vocabulary-based prompt conditioning
- Corrections logged for user review (`corrections.log`)
- Manual override: user can mark corrections as incorrect
**Files:**
- `src/clip_video/vocabulary/terms.py`
- `src/clip_video/vocabulary/correction.py`
- `src/clip_video/vocabulary/phonetic.py`
- `src/clip_video/vocabulary/cncf_starter.json`
- `tests/test_vocabulary.py`

### 7. Whisper transcription integration
**Depends on:** 5, 6
**Description:** Integrate with Whisper for transcription. Support both OpenAI Whisper API and local whisper.cpp/faster-whisper. Implement provider abstraction so user can configure preferred provider. Include cost estimation before processing (API mode). Store progress to enable resume if interrupted. Use brand vocabulary for prompt conditioning and post-transcription correction.
**Acceptance:**
- `clip-video transcribe <brand>` processes all untranscribed videos
- Cost estimate shown before API transcription begins
- Resume works after interruption (doesn't re-process completed videos)
- Word-level timestamps captured where available
- Provider configurable via brand config or CLI flag
- Vocabulary-based corrections applied automatically
**Files:**
- `src/clip_video/transcription/base.py`
- `src/clip_video/transcription/whisper_api.py`
- `src/clip_video/transcription/whisper_local.py`
- `src/clip_video/transcription/progress.py`
- `tests/test_transcription.py`

### 8. Idempotent processing and progress tracking
**Depends on:** 3, 7
**Description:** Implement processing state machine that tracks which videos have been transcribed, which clips extracted, etc. Ensure all operations are idempotent - running the same command twice produces same result without duplicate work. Add progress reporting to CLI with ETA.
**Acceptance:**
- Re-running transcription skips already-processed videos
- Progress file tracks completed items
- CLI shows progress bar with ETA for batch operations
- Interrupted operations resume cleanly
- State file uses atomic writes
**Files:**
- `src/clip_video/progress.py`
- `src/clip_video/state.py`

### 9. Lyrics parser and phrase extraction
**Depends on:** 5
**Description:** Parse lyrics text file into lines and words. Support manual phrase boundary markers (e.g., `[phrase]word word word[/phrase]`). Generate word/phrase list from lyrics that drives dictionary building. Handle common lyrics formatting (verse markers, repeated sections).
**Acceptance:**
- Parse plain text lyrics into word list
- Support phrase markers for multi-word extraction
- Deduplicate words/phrases in extraction list
- Handle common lyrics file formats
**Files:**
- `src/clip_video/lyrics/parser.py`
- `src/clip_video/lyrics/phrases.py`
- `tests/test_lyrics.py`

### 10. Phrase search across brand library
**Depends on:** 5, 9
**Description:** Implement search command that finds all occurrences of a word/phrase across all videos in a brand. Rank results by match quality. Output candidate clips to a search results folder for review. Track which clips have been selected to avoid reuse.
**Acceptance:**
- `clip-video search <brand> "phrase"` finds all matches
- Results show source video, timestamp, and context
- Candidates exported as preview clips to search_results folder
- Selection tracking prevents same source video reuse in one project
**Files:**
- `src/clip_video/search.py`
- `src/clip_video/selection.py`
- `tests/test_search.py`

### 11. Lyric match clip extraction
**Depends on:** 4, 9, 10
**Description:** Implement the full lyric match workflow: parse lyrics, search for each word/phrase, extract candidate clips, output organized by lyric line. Include CLI command to start a lyric match project and process iteratively.
**Acceptance:**
- `clip-video lyric-match <brand> <project> <lyrics-file>` starts project
- Each lyric line gets folder with candidate clips
- Clips named `{source_video}_{timestamp}.mp4`
- Progress saved; can resume interrupted extraction
- Summary report shows coverage gaps
**Files:**
- `src/clip_video/modes/lyric_match.py`
- `tests/test_lyric_match.py`

### 12. LLM integration for highlight detection
**Depends on:** 5
**Description:** Create LLM client abstraction supporting Claude and OpenAI APIs. Design prompt that takes transcript + session description and identifies top 3-5 highlight-worthy segments. Return structured response with timestamps, reasoning, and suggested hook text.
**Acceptance:**
- LLM analyzes transcript and returns highlight segments
- Each highlight has: start/end times, summary, hook text
- Supports both Claude and OpenAI APIs
- Cost estimation available before processing
- Handles rate limits with backoff
**Files:**
- `src/clip_video/llm/base.py`
- `src/clip_video/llm/claude.py`
- `src/clip_video/llm/openai.py`
- `src/clip_video/llm/prompts.py`
- `tests/test_llm.py`

### 13. Caption rendering with FFmpeg
**Depends on:** 4, 5
**Description:** Implement caption burning using FFmpeg drawtext filter or ASS subtitles. Position captions in lower third of frame. Auto-size text to fit video width with word wrapping. Support configurable font, size, and colors.
**Acceptance:**
- Captions burn into video at correct timestamps
- Text positioned in lower third
- Long text wraps appropriately
- Font/size/color configurable
- Output compatible with portrait and landscape formats
**Files:**
- `src/clip_video/captions/renderer.py`
- `src/clip_video/captions/styles.py`
- `tests/test_captions.py`

### 14. Brand emoji and logo injection
**Depends on:** 13
**Description:** Extend caption renderer to inject brand-specific emojis or logos when configured trigger words are spoken. Load mappings from brand config. Support both Unicode emoji (in text) and image overlays (logos).
**Acceptance:**
- Emoji appears in caption when trigger word spoken
- Logo image overlays at configured position
- Mappings loaded from brand config.json
- Multiple triggers can fire in same caption
**Files:**
- `src/clip_video/captions/enhancements.py`
- `tests/test_enhancements.py`

### 15. Portrait crop and format conversion
**Depends on:** 4
**Description:** Implement landscape to portrait (9:16) conversion using fixed crop region from brand config. Maintain quality settings appropriate for YouTube Shorts / LinkedIn. Output as MP4 with web-optimized settings.
**Acceptance:**
- Landscape video converts to 9:16 portrait
- Crop region configurable per brand
- Output optimized for social platforms
- Quality acceptable for YouTube Shorts upload
**Files:**
- `src/clip_video/video/portrait.py`
- `tests/test_portrait.py`

### 16. Highlights mode full pipeline
**Depends on:** 12, 13, 14, 15
**Description:** Implement complete highlights workflow: load video + description, transcribe if needed, send to LLM for analysis, extract highlight clips, convert to portrait, burn captions with enhancements, generate metadata. Output organized clips ready for upload.
**Acceptance:**
- `clip-video highlights <brand> <video> --description <file>` processes single video
- Outputs 3-5 highlight clips in portrait format
- Each clip has burned-in captions with brand enhancements
- Metadata file with title/hook text for each platform
- Progress saved for resume capability
**Files:**
- `src/clip_video/modes/highlights.py`
- `tests/test_highlights.py`

### 17. Batch processing orchestration
**Depends on:** 8, 16
**Description:** Implement batch processing for highlights mode that handles 30+ videos. Process in parallel where possible. Track progress across all videos. Generate summary report with all outputs and any failures.
**Acceptance:**
- `clip-video highlights-batch <brand> <video-list>` processes multiple videos
- Progress shown across entire batch
- Failed videos logged but don't stop batch
- Summary report lists all generated clips
- Can resume interrupted batch
**Files:**
- `src/clip_video/batch.py`
- `tests/test_batch.py`

### 18. Error recovery and logging
**Depends on:** 8
**Description:** Implement comprehensive error handling with retry logic for transient failures (network, API rate limits). Add structured logging with configurable verbosity. Ensure errors never corrupt state files.
**Acceptance:**
- Network failures retry with exponential backoff
- API rate limits handled with wait and retry
- All errors logged with context
- `--verbose` flag increases log detail
- State files never corrupted by errors
**Files:**
- `src/clip_video/errors.py`
- `src/clip_video/logging.py`

### 19. Cost reporting and summaries
**Depends on:** 7, 12, 17
**Description:** Track API costs (transcription, LLM) throughout processing. Generate cost summary at end of operations. Include time estimates for future similar batches.
**Acceptance:**
- Running cost total shown during processing
- Final summary shows total API spend
- Cost breakdown by service (transcription vs LLM)
- Estimate provided before batch starts
**Files:**
- `src/clip_video/costs.py`
- `tests/test_costs.py`

---

## Dependencies Graph

```
1 (scaffolding)
├── 2 (ffmpeg binary) ─── 4 (ffmpeg wrapper) ─┬── 13 (captions) ─── 14 (emoji) ─┐
│                                              └── 15 (portrait) ────────────────┤
│                                                                                │
├── 3 (data models) ─┬── 5 (transcript) ─┬── 7 (whisper) ───┐                   │
│                    │                    ├── 9 (lyrics) ────┼── 10 (search) ── 11 (lyric-match)
│                    │                    └── 12 (llm) ──────┤                   │
│                    │                                       │                   │
│                    ├── 6 (vocabulary) ─── 7 (whisper)      │                   │
│                    │                                       │                   │
│                    └── 8 (idempotent) ─────────────────────┼── 17 (batch) ────┤
│                                                            │                   │
│                                                            └───────────────── 16 (highlights)
│
└── 18 (error handling) ←── 8
    19 (cost reporting) ←── 7, 12, 17
```

---

## Execution Notes

- **Phase 1 (Tasks 1-5):** Foundation - scaffolding, FFmpeg, data models, transcripts
- **Phase 2 (Tasks 6-8):** Transcription - vocabulary correction, Whisper integration, idempotency
- **Phase 3 (Tasks 9-11):** Lyric Match Mode - independent of highlights
- **Phase 4 (Tasks 12-16):** Highlights Mode - independent of lyric match
- **Phase 5 (Tasks 17-19):** Polish - batch processing, error handling, cost reporting

Critical path: 1 → 2 → 4 → 3 → 5 → 7 → 12 → 16 → 17

---

## Risk Areas

1. **FFmpeg binary distribution** - Platform-specific binaries, licensing considerations
2. **FFmpeg complexity** - Caption burning and portrait crop require complex filter chains
3. **Whisper accuracy** - Word-level timestamps may be unreliable; vocabulary correction helps but won't catch everything
4. **Vocabulary coverage** - Technical terms beyond CNCF may need user-added entries
5. **LLM prompt engineering** - Highlight detection quality depends on prompt design
6. **API costs** - Need accurate estimation to stay under $20/conference threshold

---

## Starter CNCF Vocabulary

Included in task 6, the starter vocabulary covers common CNCF terms:

```json
{
  "vocabulary": {
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
    "cncf": ["c n c f", "cnc f", "see ncf"]
  }
}
```

Users can extend this per-brand with conference-specific terms.
