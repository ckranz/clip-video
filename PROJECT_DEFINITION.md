# Project: Clip Video - Intelligent Video Clip Extraction Tool

## Goal

A unified tool for intelligent video clip extraction with two primary modes:

1. **Lyric Match Mode**: Find clips where spoken words match target song lyrics for music video mashups
2. **Highlights Mode**: Automatically identify and extract compelling "shorts" from longer video content

## Core Workflow

### Shared Pipeline

1. **Ingest**: Video file(s) → transcript generation/import with timestamps
2. **Analysis**: LLM-powered content understanding
3. **Extraction**: FFmpeg clip extraction with configurable padding
4. **Output**: Organized clips with metadata, ready for editing

### Mode-Specific Processing

#### Lyric Match Mode

- Input: Song lyrics + source video(s)
- Match lyrics to transcript segments (phonetic/semantic)
- Output: Clips tagged by lyric line

#### Highlights Mode

- Input: Video + topic preferences/interests (configurable profile)
- LLM analyzes transcript for: key insights, quotable moments, technical explanations, humor, surprising statements
- Filters by user-specified topics of interest
- Ranks and selects top 3-5 segments
- Respects "shorts" constraints (30-60 sec, standalone context)
- Output: Ready-to-post clips with suggested titles/hooks

## Key Components

### Transcript Engine (shared)

- Generate via Whisper (local/API) or import VTT/SRT
- Word-level timestamps where possible
- Handle technical vocabulary better than generic tools

### Content Analyzer (shared, configurable)

- LLM-powered segment scoring
- User-defined topic profiles and interests
- Technical term recognition and proper handling

### Lyric Matcher (mode 1)

- Phonetic matching (Soundex, Metaphone)
- Semantic similarity for creative matches
- Match quality scoring

### Highlight Detector (mode 2)

- Identify self-contained interesting moments
- Score by: relevance to interests, quotability, engagement potential
- Ensure clips have proper context (no mid-thought cuts)

### Clip Extractor (shared)

- FFmpeg-based extraction
- Configurable handles/padding
- Batch processing

### Project/Profile Management (shared)

- Save user topic profiles and preferences
- Project state persistence
- Export formats for different platforms

## Technical Stack

- **Runtime**: Node.js or Python (TBD based on library ecosystem)
- **Video Processing**: FFmpeg
- **Transcription**: Whisper (local via ollama or API)
- **AI Analysis**:
  - Local ollama for bulk transcript processing
  - Claude/GPT API for nuanced matching and highlight detection
- **Storage**: Local JSON/SQLite

## Advantages Over Existing Tools (e.g., Opus Clip)

- **Personalized interests**: Trained on topics you care about, not generic engagement metrics
- **Technical accuracy**: Better handling of domain-specific vocabulary
- **Transparency**: See why clips were selected, adjust criteria
- **Local control**: Content stays local, no upload limits or privacy concerns

## Out of Scope (v1)

- Auto-generated captions/subtitles on output
- Direct social media posting
- Video resizing/formatting for platforms
- Automatic final video assembly (manual editing still required for lyric mode)
- Real-time preview
- Mouth movement/viseme analysis

## Success Criteria

- **Lyric mode**: 70%+ reduction in clip-finding time compared to manual scrubbing
- **Highlights mode**: Generates clips more relevant to user interests than Opus Clip
- **Performance**: Processes 1-hour video in under 10 minutes
- **Usability**: Clear quality indicators and reasoning for all selections
- **Scale**: Can process a 3-minute song against 1+ hours of source footage
