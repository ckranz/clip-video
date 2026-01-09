# Discovery Document: Clip Video - Intelligent Video Clip Extraction Tool

**Session ID:** clip-video-extraction-tool-20260109
**Date:** 2026-01-09
**Status:** Ready for Implementation
**Exchanges:** 7 | **Discoveries:** 41

---

## Executive Summary

A Python CLI tool for intelligent video clip extraction with two primary modes:

1. **Lyric Match Mode**: Build searchable word/phrase dictionaries from conference video libraries, enabling rapid creation of music video mashups where speakers "sing" songs
2. **Highlights Mode**: Batch-process conference recordings to automatically generate social media shorts with burned-in captions and brand-specific visual enhancements

Target user: Single user (tool author) processing 15-20 hours of conference content per event.

---

## Core Requirements

### Mode 1: Lyric Match Mode

| Requirement | Detail |
|-------------|--------|
| **Input** | Folder of source videos + target song lyrics |
| **Dictionary** | Word/phrase clips organized by "brand" (conference), persists across song projects |
| **Phrase Handling** | Manually curated boundaries, lyrics-driven extraction |
| **Search** | CLI search interface: "find phrase X in brand Y" |
| **Selection** | Human-in-the-loop; phonetic matching is nice-to-have |
| **Output** | Clips named `{source_video}_{clip_id}` dumped to folder |
| **Constraint** | Prefer not reusing clips from same source video in one output |

### Mode 2: Highlights Mode

| Requirement | Detail |
|-------------|--------|
| **Input** | Source video + session description/abstract |
| **Batch Scale** | 30+ videos per conference event |
| **Target Platforms** | YouTube Shorts, LinkedIn (portrait 9:16) |
| **Selection Goals** | (1) Hook into full video, OR (2) standalone knowledge snippet |
| **Output** | Video clip + title/hook text for each platform |
| **Captions** | Burned-in, lower-thirds, auto-sized/wrapped |
| **Brand Enhancement** | Inject emojis/logos when specific terms mentioned |
| **Portrait Crop** | Fixed center crop (speakers typically stationary) |

### Shared Requirements

| Requirement | Detail |
|-------------|--------|
| **Interface** | CLI only - no web UI |
| **State** | Save/resume project state; idempotent processing |
| **Reliability** | Failure-tolerant for unreliable internet; no progress loss |
| **Transcripts** | Generate via Whisper (local/API) or import existing VTT/SRT |
| **Retention** | Keep full transcripts and source videos for future reuse |
| **Storage** | Videos in `brand/source/`; clips in separate output folders |

---

## Technical Decisions

### Runtime & Environment

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Python | Stronger ecosystem for video/ML tooling |
| **Target OS** | Windows 11 | User's primary machine |
| **Hardware** | Surface Book Pro, 32GB RAM, laptop GPU | Limits local Whisper; may favor API |

### External Services

| Service | Provider | Constraint |
|---------|----------|------------|
| **Transcription** | Whisper (local or API) | Prefer cheap; ~$20 threshold for full conference |
| **LLM Analysis** | Claude or OpenAI API | For highlight detection |
| **Cost Estimation** | Required before processing | Make provider configurable |

### Video Processing

| Aspect | Decision |
|--------|----------|
| **Tool** | FFmpeg |
| **Container** | MP4 or MOV (DaVinci Resolve compatible) |
| **Video Codec** | H.264 or H.265 |
| **Audio Codec** | AAC |
| **Resolution** | Standard (no 4K needed) |
| **Portrait Conversion** | Fixed crop, not visual tracking |

### Caption Rendering

| Aspect | Decision |
|--------|----------|
| **Method** | Direct text overlay (burned in) |
| **Position** | Lower thirds |
| **Sizing** | Auto-sized and auto-wrapped |
| **Enhancement** | Brand-specific emoji/logo injection |
| **Nice-to-have** | Word highlighting animation (karaoke-style) |

---

## Data Model

### Brand Library
```
brand/
├── config.json           # Brand settings, emoji mappings
├── source/              # Original video files
├── transcripts/         # Generated/imported transcripts
└── clips/               # Pre-cut word/phrase clips
    └── {word_or_phrase}/
        └── {source_video}_{timestamp}.mp4
```

### Project (Song/Highlights)
```
project/
├── project.json         # Project state, progress tracking
├── input/              # Lyrics file or video list
├── output/             # Generated clips
│   ├── clips/          # Video files
│   └── metadata/       # Titles, descriptions per clip
└── search_results/     # Temporary search outputs
```

### Brand Configuration
```json
{
  "name": "KubeCon 2025",
  "emoji_mappings": {
    "kubernetes": "☸️",
    "docker": "🐳",
    "cloud": "☁️"
  },
  "logo_mappings": {
    "kubernetes": "assets/k8s-logo.png"
  },
  "default_crop": {"x": 0.25, "y": 0, "width": 0.5, "height": 1.0}
}
```

---

## Processing Pipeline

### Transcription Phase
1. Scan source folder for videos
2. Check for existing transcripts (skip if present - idempotent)
3. Estimate cost/time for remaining videos
4. Process transcription (resumable at video level)
5. Store transcripts with word-level timestamps

### Lyric Match Pipeline
1. Parse target lyrics into words/phrases
2. For each term, search transcript index across brand
3. Extract candidate clips with configurable padding
4. Output candidates to folder for manual review
5. Track selected clips to avoid source reuse

### Highlights Pipeline
1. Load video + session description
2. Send transcript + description to LLM for analysis
3. LLM identifies top 3-5 highlight segments
4. For each highlight:
   - Extract clip with context boundaries
   - Convert to portrait (fixed crop)
   - Burn in captions with brand enhancements
   - Generate title/hook text
5. Output clips + metadata to folder

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| **Transcription interrupted** | Resume from last completed video |
| **Network failure** | Retry with backoff; never corrupt state |
| **API rate limit** | Pause and wait; estimate queue position |
| **Invalid video file** | Log error, skip, continue batch |
| **Insufficient matches** | Report gaps, suggest alternatives |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| **Lyric mode time savings** | 70%+ reduction vs manual scrubbing |
| **Highlights relevance** | More topic-aware than Opus Clip |
| **Processing speed** | 1-hour video in <10 minutes |
| **Conference batch** | 15-20 hours processable overnight |
| **Cost per conference** | Under $20 for full transcription |

---

## Out of Scope (v1)

- Direct social media posting
- Video resizing beyond fixed crop
- Real-time preview
- Mouth movement/viseme analysis
- Auto-generated subtitles on final output (captions are burned in, not sidecar)
- Multi-user or cloud deployment

---

## Open Questions / Future Considerations

1. **Phonetic matching algorithm**: Soundex vs Metaphone vs embedding similarity - defer to implementation
2. **Transcript format**: Store as JSON with word timestamps, exportable to SRT/VTT
3. **Clip padding**: Make configurable per project (suggested default: 0.3s before, 0.5s after)
4. **DaVinci Resolve integration**: Consider generating EDL/XML for timeline import (v2)

---

## Recommended Implementation Order

### Phase 1: Foundation
1. Project/brand data structures and CLI scaffolding
2. FFmpeg wrapper for basic clip extraction
3. Transcript storage and indexing

### Phase 2: Transcription
4. Whisper integration (API first, local optional)
5. Cost estimation and progress tracking
6. Idempotent resume capability

### Phase 3: Lyric Match Mode
7. Lyrics parser and phrase search
8. Clip extraction with padding
9. Search interface for sourcing phrases

### Phase 4: Highlights Mode
10. LLM integration for highlight detection
11. Portrait crop and caption burning
12. Brand emoji/logo injection

### Phase 5: Polish
13. Batch processing orchestration
14. Error recovery and logging
15. Cost reporting and summaries
