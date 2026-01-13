# Clip Video

Intelligent video clip extraction tool with two primary modes:

1. **Highlights Mode**: Generate social media shorts from conference/podcast recordings with AI-detected highlights and burned-in captions
2. **Lyric Match Mode**: Build searchable word/phrase dictionaries for creating music video mashups where speakers "sing" songs

## Installation

```bash
pip install -e .
```

### Requirements

- **Python 3.10+**
- **FFmpeg** - Required for video processing (auto-downloaded on first run, or install manually)
- **API Keys** (set as environment variables or in `.env` file):
  - `OPENAI_API_KEY` - For Whisper transcription (required)
  - `ANTHROPIC_API_KEY` - For Claude-based highlight detection (recommended)

---

## Quick Start: Highlights Mode

Generate social media shorts from recordings with burned-in captions.

### 1. Set up a brand and add videos

```bash
# Create brand
clip-video init-brand "TechConf2024" --description "Tech Conference 2024"

# Copy videos to the brand folder
cp /path/to/videos/*.mp4 brands/TechConf2024/videos/
```

### 2. Transcribe videos

```bash
# Transcribe all videos (shows cost estimate first)
clip-video transcribe TechConf2024

# Or skip confirmation for batch runs
clip-video transcribe TechConf2024 --yes
```

### 3. Generate highlights

```bash
# Generate 5 highlight clips from a video
clip-video highlights TechConf2024 "keynote.mp4"

# Generate more highlights
clip-video highlights TechConf2024 "keynote.mp4" --count 10
```

Output:

```
brands/TechConf2024/highlights/keynote_<timestamp>/
├── clips/
│   ├── landscape/       # Original 16:9 clips
│   ├── portrait/        # Cropped 9:16 for shorts
│   └── captioned/       # Portrait with burned-in captions
├── project_state.json
└── report.txt
```

### Batch Processing

Process multiple videos unattended:

```bash
# Create list of videos
echo "session1.mp4" > videos.txt
echo "session2.mp4" >> videos.txt

# Batch process (resumes if interrupted)
clip-video highlights-batch TechConf2024 videos.txt --count 5
```

---

## Quick Start: Lyric Match Mode

Helper for creating music video and video clip mashups where speakers "sing" songs or say something new.

### 1. Set up brand and transcribe (same as above)

```bash
clip-video init-brand "TechConf2024"
# Add videos to brands/TechConf2024/videos/
clip-video transcribe TechConf2024 --yes
```

### 2. Check what words are available

Before writing lyrics, check what words exist in your transcripts:

```bash
# Export all unique words to a dictionary file
clip-video export-dictionary TechConf2024

# Check specific lyrics against available words
clip-video check-lyrics TechConf2024 my_lyrics.txt
```

### 3. Write lyrics file

Create a text file with your parody lyrics:

```text
# my_lyrics.txt
[Verse 1]
I was working on the cloud
You made an engineer out of me

[Chorus]
I'm moving on up now
Getting out of the monolith
```

### 4. Extract clips

```bash
# Create project and extract clips
clip-video lyric-match TechConf2024 "my-parody" my_lyrics.txt

# Update with modified lyrics (keeps existing clips)
clip-video lyric-match TechConf2024 "my-parody" my_lyrics.txt --update

# Resume extraction (uses original lyrics)
clip-video lyric-match TechConf2024 "my-parody" my_lyrics.txt --resume
```

Output shows **word coverage** as the critical metric:

```
╭──────────────────── Results ────────────────────╮
│ Lyric Match Complete!                           │
│                                                 │
│ ✓ Word coverage: 25/25 (100%)                   │
│ Phrase coverage: 45/120                         │
│ Total clips extracted: 150                      │
│                                                 │
│ Output folder: .../projects/my-parody/clips     │
╰─────────────────────────────────────────────────╯
```

Clips are organized by word/phrase:

```
brands/TechConf2024/projects/my-parody/clips/
├── cloud/
│   ├── video1_45.2s.mp4
│   └── video2_123.5s.mp4
├── engineer/
│   ├── video1_67.8s.mp4
│   └── video3_89.1s.mp4
└── moving/
    └── ...
```

---

## Configuration

### Brand Configuration

Each brand has a `config.json` file. See `brands/example/config.json` for a full example.

Key settings:

```json
{
  "name": "TechConf2024",
  "description": "Tech Conference 2024",

  "portrait": {
    "crop_x_offset": 0.5,
    "crop_x_pixels": null,
    "crop_source_width": null
  },

  "logo": {
    "enabled": false,
    "image_path": "logo/logo.png",
    "position": "top-center"
  },

  "vocabulary": {
    "kubernetes": ["cooper netties", "kuber nettis"],
    "istio": ["is theo", "east io"]
  }
}
```

**Vocabulary**: Maps correct spellings to common Whisper mistranscriptions. The tool searches for both the correct word and its alternatives.

**Portrait crop**: Control how landscape videos are cropped to portrait:

- `crop_x_offset`: 0.0 (left) to 1.0 (right), 0.5 = center
- `crop_x_pixels`: Exact pixel position (overrides offset)

### API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

---

## Commands Reference

### Brand Management

| Command           | Description                              |
| ----------------- | ---------------------------------------- |
| `init-brand NAME` | Create a new brand with folder structure |
| `list-brands-cmd` | List all brands                          |
| `info BRAND`      | Show brand details and statistics        |

### Transcription

| Command                         | Description                      |
| ------------------------------- | -------------------------------- |
| `transcribe BRAND`              | Transcribe all videos in a brand |
| `transcribe BRAND --video FILE` | Transcribe specific video        |
| `transcribe BRAND --force`      | Re-transcribe even if exists     |
| `index-transcripts BRAND`       | Rebuild search index             |

### Highlights Mode

| Command                        | Description                    |
| ------------------------------ | ------------------------------ |
| `highlights BRAND VIDEO`       | Generate highlights from video |
| `highlights BRAND VIDEO -n 10` | Generate 10 highlights         |
| `highlights-batch BRAND LIST`  | Batch process video list       |

### Lyric Match Mode

| Command                       | Description                    |
| ----------------------------- | ------------------------------ |
| `export-dictionary BRAND`     | Export all words to file       |
| `check-lyrics BRAND FILE`     | Check word availability        |
| `search BRAND "word"`         | Search for word occurrences    |
| `lyric-match BRAND NAME FILE` | Create/run lyric match project |
| `lyric-match ... --update`    | Update with new lyrics         |
| `lyric-match ... --resume`    | Resume existing project        |

### Utilities

| Command        | Description                   |
| -------------- | ----------------------------- |
| `check-deps`   | Check FFmpeg and dependencies |
| `review BRAND` | Manage rejected clips queue   |

---

## Costs

### Transcription (OpenAI Whisper API)

- ~$0.006 per minute of audio
- 30-minute video ≈ $0.18

### Highlight Detection (Claude API)

- ~$0.05-0.15 per video analysis
- Depends on transcript length

### Total per video (30 min)

- First run: ~$0.20-0.35 (transcription + analysis)
- Subsequent: ~$0.05-0.15 (analysis only, transcript reused)

---

## Tips

1. **Transcribe first**: Run transcription separately - it's the main cost, and is reusable across modes.

2. **Check words before writing lyrics**: Use `check-lyrics` or `export-dictionary` to see what words are available in your corpus.

3. **Word coverage matters most**: For lyric match, focus on 100% word coverage. Phrases are nice-to-have but you can always assemble them from individual words.

4. **Vocabulary helps**: Add domain-specific terms to your brand config to catch Whisper mistranscriptions like "Kubernetes" → "Cooper Netties".

5. **Resume-safe**: All batch operations can be interrupted and resumed.

---

## Troubleshooting

### FFmpeg not found

```bash
clip-video check-deps
# FFmpeg will be auto-downloaded on first video operation
```

### Missing words in lyric match

- Check `coverage_report.md` for missing words
- Add more source videos, or modify lyrics to use available words
- Some words may need manual recording

### Transcription errors

- Add terms to brand vocabulary config
- The tool searches for both correct spellings and known alternatives

### API rate limits

- The tool automatically retries with backoff
- Monitor usage in OpenAI/Anthropic dashboards
